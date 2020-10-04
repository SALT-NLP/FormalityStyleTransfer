# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.trans_only_updates = args.trans_only_epoch * 919
        self.sentence_avg = args.sentence_avg
        self.eps = args.label_smoothing
        self.weight_forward = args.weight_forward
        self.cycle_weight = args.cycle_weight
        self.recon_weight = args.recon_weight
        self.disc_weight = args.disc_weight
 
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        # fmt: on

    def forward(self, model, sample, num_updates,reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        cycle_weight = self.cycle_weight #if ((self.trans_only_updates < num_updates) and 'train' in sample) else 0.0
        recon_weight = self.recon_weight #if ((self.trans_only_updates < num_updates) and 'train' in sample) else 0.0
        disc_weight = self.disc_weight
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tot_nll_loss = torch.zeros(1,dtype = torch.double, requires_grad = True).to(device)
        tot_trans_loss = torch.zeros(1,dtype = torch.double, requires_grad = True).to(device)
        tot_disc_loss = torch.zeros(1,dtype = torch.double, requires_grad = True).to(device)
        tot_recon_loss = torch.zeros(1,dtype = torch.double, requires_grad = True).to(device)
        tot_cycle_loss = torch.zeros(1,dtype = torch.double, requires_grad = True).to(device)
        accuracy =torch.zeros(1,dtype = torch.double).to(device)
        tot_sz = 0
        forward_weight = [self.weight_forward, 1.0 - self.weight_forward]

        if (True):
            for inp_tokens, inp_length, label, label_input, prev_output_tokens, target in zip(
                [sample['net_input']['src_tokens'], sample['net_target']['src_tokens']],
                [sample['net_input']['src_lengths'], sample['net_target']['src_lengths']],
                [1.0,0.0],
                [0.0,1.0],
                [sample['net_input']['prev_output_tokens'], sample['net_target']['prev_output_tokens']],
                [sample['net_target']['src_tokens'], sample['net_input']['src_tokens']],

            ):
                decoder_out, extra, logits = model(inp_tokens,inp_length,prev_output_tokens, tgt_lang = label, classification_head_name = 'formal')
                labels = (torch.ones(inp_tokens.size(0), dtype=torch.int64, requires_grad=False) * int(label)).to(device)
                pred = torch.argmax(logits, axis = 1)
                accuracy = accuracy +  torch.sum(pred==label).double()
                disc_loss = F.nll_loss(F.log_softmax(logits, dim=-1, dtype=torch.float32),labels,reduction='sum')
                loss, nll_loss = self.compute_loss(model, (decoder_out, extra), target, reduce=reduce)

                tot_disc_loss = tot_disc_loss + forward_weight[int(label_input)] * disc_loss
                tot_sz = pred.size(0) + tot_sz
                tot_trans_loss = tot_trans_loss + forward_weight[int(label_input)] * loss
                if (label==1.0):
                    tot_nll_loss = tot_nll_loss + nll_loss


        sampled = sample['plain']
        if (recon_weight + cycle_weight > 0):
            for inp_tokens, inp_length, label, label_output,  prev_output_tokens in zip(
                [sampled['net_input']['src_tokens'], sampled['net_target']['src_tokens']],
                [sampled['net_input']['src_lengths'], sampled['net_target']['src_lengths']],
                [0.0,1.0],
                [1.0,0.0],
                [sampled['net_target']['prev_output_tokens'], sampled['net_input']['prev_output_tokens']],
            ):
                if (recon_weight > 0):
                    decoder_out, extra, logits = model(inp_tokens,inp_length,prev_output_tokens, tgt_lang = label)
                    loss, _ = self.compute_loss(model, (decoder_out, extra), inp_tokens, reduce=reduce)
                    tot_recon_loss = tot_recon_loss + forward_weight[int(label)] * loss
                if (cycle_weight > 0):
                    decoder_out, extra, logits = model(inp_tokens,inp_length, prev_output_tokens, tgt_lang = label_output)
                    src_tokens= torch.argmax(decoder_out, 2)
                    src_lengths = torch.ones(src_tokens.size()[0])
                    for i in range(0, src_tokens.size()[0]):
                        for j in range(src_tokens.size()[1]-1,0,-1):
                            if (src_tokens[i][j] != 2):
                                src_lengths[i] = j + 1
                                break
                    decoder_out, extra, _ = model(src_tokens,src_lengths,prev_output_tokens, tgt_lang = label)
                    loss, _ = self.compute_loss(model, (decoder_out, extra), inp_tokens, reduce=reduce)
                    tot_cycle_loss = tot_cycle_loss + forward_weight[int(label)] * loss



        sample_size = sample['net_target']['src_tokens'].size(0) if self.sentence_avg else sample['ntokens']
        accuracy = accuracy/tot_sz
        tot_loss = 2.0 * (tot_trans_loss + disc_weight*tot_disc_loss + recon_weight*tot_recon_loss + cycle_weight*tot_cycle_loss)
        logging_output = {
            'loss': tot_loss.data,
            'trans_loss': tot_trans_loss.data,
            'cycle_loss': tot_cycle_loss.data,
            'recon_loss': tot_recon_loss.data,
            'accuracy': accuracy.data,
            'disc_loss':tot_disc_loss.data,
            'nll_loss':tot_nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['net_target']['src_tokens'].size(0),
            'sample_size': sample_size,
        }
        return tot_loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        trans_loss_sum = utils.item(sum(log.get('trans_loss', 0) for log in logging_outputs))
        recon_loss_sum = utils.item(sum(log.get('recon_loss', 0) for log in logging_outputs))
        cycle_loss_sum = utils.item(sum(log.get('cycle_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        disc_loss_sum = utils.item(sum(log.get('disc_loss', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        accuracy_sum = utils.item(sum(log.get('accuracy', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))

        metrics.log_scalar('accuracy', accuracy_sum / len(logging_outputs), len(logging_outputs), round=3)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('trans_loss', trans_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('disc_loss', disc_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('cycle_loss', cycle_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('recon_loss', recon_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
