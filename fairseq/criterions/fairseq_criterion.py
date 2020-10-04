# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Dict, List

from torch.nn.modules.loss import _Loss

from fairseq import metrics, utils


class FairseqCriterion(_Loss):

    def __init__(self, task):
        super().__init__()
        self.task = task
        if hasattr(task, 'target_dictionary'):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(
        logging_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            'The aggregate_logging_outputs API is deprecated. '
            'Please use the reduce_metrics API instead.'
        )
        raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            'Criterions should implement the reduce_metrics API. '
            'Falling back to deprecated aggregate_logging_outputs API.'
        )
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {'nsentences', 'ntokens', 'sample_size'}:
                continue
            metrics.log_scalar(k, v)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


class LegacyFairseqCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task=task)
        self.args = args

        utils.deprecation_warning(
            'Criterions should take explicit arguments instead of an '
            'argparse.Namespace object, please update your criterion by '
            'extending FairseqCriterion instead of LegacyFairseqCriterion.'
        )

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)
