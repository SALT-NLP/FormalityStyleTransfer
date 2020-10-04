# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

import math

from fairseq import utils
import torch.nn.functional as F


from . import FairseqCriterion, register_criterion
#from fairseq.data.data_utils import insert_lang_code

@register_criterion('classification')
class ClassificationCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing=0):
        super().__init__(task)
        self.eps = label_smoothing
        self.sentence_avg = sentence_avg

    # @staticmethod
    # def add_args(parser):
    #     """Add criterion-specific arguments to the parser."""
    #     parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
    #                         help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, update_num, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss = torch.zeros(1, dtype = torch.double).to(device)
        accuracy = 0.0
        if 'plain' in sample:
            sample = sample['plain']

        for targ1,targ2,targ3, label in zip(
            [sample['net_target']['src_tokens'], sample['net_input']['src_tokens']],
            [sample['net_target']['src_lengths'], sample['net_input']['src_lengths']],
            [sample['net_input']['prev_output_tokens'], sample['net_target']['prev_output_tokens']],
            [1.0, 0.0],
        ):
            _, _ , logits = model(targ1,targ2,targ3,features_only=True, classification_head_name = 'formal')
            logits = logits.reshape((-1,2))
            labels = (torch.ones(targ1.size(0), dtype=torch.int64, requires_grad=False) * int(label)).to(device)
            pred = torch.argmax(logits, axis = 1)
            discri_loss = F.nll_loss(F.log_softmax(logits, dim=-1, dtype=torch.float32),labels,reduction='sum',)
            acc = torch.sum(pred==label).double() / pred.size(0)

            loss += discri_loss
            accuracy += acc


        accuracy /= 2;

        nll_loss = (torch.ones(1, dtype = torch.double) * (11 - accuracy)).to(device)
        sample_size = sample['net_target']['src_tokens'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data),
            'accuracy': utils.item(accuracy.data),
            'ntokens': sample['ntokens'],
            'nll_loss' : utils.item(nll_loss.data),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / len(logging_outputs),
            'accuracy': sum(log.get('accuracy', 0) for log in logging_outputs) / len(logging_outputs),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / len(logging_outputs) ,
            # 'discri_loss': sum(log.get('discri_loss', 0) for log in logging_outputs) / len(logging_outputs),
            'sample_size': sample_size,
        }
