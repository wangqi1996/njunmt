from typing import List

import torch
import torch.nn as nn

from src.data.vocabulary import PAD
from src.utils.logging import INFO


class Criterion(nn.Module):
    """ Class for managing loss computation.

    """

    def _compute_loss(self, inputs, labels, **kwargs):
        """
        Compute the loss. Subclass must override this method.

        Args:
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        Returns:
            A non-reduced FloatTensor with shape (batch, )
        """
        raise NotImplementedError

    def forward(self, inputs, labels=None, normalization=1.0, reduce=True, seqs_x=None, y_inp=None, **kwargs):
        """
        Compute loss given inputs and labels.

        Args:
            inputs: Input tensor of the criterion.
            labels: Label tensor of the criterion.
            reduce: Boolean value indicate whether the criterion should reduce the loss along the batch. If false,
                the criterion return a FloatTensor with shape (batch, ), otherwise a scalar.
            normalization: Normalization factor of the loss. Should be a float scalar or a FloatTensor with shape
                (batch, )
        """
        loss = self._compute_loss(inputs, labels, **kwargs).div(normalization)  # (batch, )

        if reduce:
            loss = loss.sum()

        return loss

    def __str__(self):
        return "criterion"


class NMTCriterion(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self, padding_idx=PAD, label_smoothing=0.0):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:

            self.criterion = nn.KLDivLoss(size_average=False, reduce=False)

        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=padding_idx, reduce=False)

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, inputs, labels, **kwargs):

        """
        Args:
            inputs (..., K): Expect logarithm probabilities.
            labels (...,): Index tensor. Should be the same size as inputs except the last dimension.
        """

        batch_size = labels.size(0)

        scores = self._bottle(inputs)  # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()  # mask of PAD

            one_hot = self._smooth_label(num_tokens)  # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()

        loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)

        return loss

    def __str__(self):
        return "nmt_criterion"


class WordKDCriterion(Criterion):
    """
    use word-level knowledge distillation loss function
    """

    def __init__(self, teacher=None, init_use_KD=False):
        super().__init__()
        self.teacher = teacher
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.use_KD_loss = init_use_KD

    def reset_teacher(self, teacher=None):
        self.teacher = teacher

    def _compute_loss(self, inputs, labels, **kwargs):
        batch_size = labels.size(0)
        loss = self.criterion(inputs, labels).view((batch_size, -1)).sum(-1)
        return loss

    def get_teacher_output(self, seqs_x, y_inp):
        assert self.teacher is not None, u"must have a teacher when calculate KD"
        output = self.teacher(seqs_x, y_inp, log_probs=False)
        return output

    def __str__(self):
        return "wordKD_criterion"


class CombinationCriterion(Criterion):
    """
    use multi loss function for train
    """

    def __init__(self, loss_config: List, **kwargs):
        """ init different loss class """
        super().__init__()
        self.nmt_criterion = None
        self.wordKD_criterion = None

        for loss in loss_config:
            type = loss['type']
            if type == "nmt_criterion":
                label_smoothing = loss.get('label_smoothing', 0.0)
                self.nmt_criterion = NMTCriterion(label_smoothing=label_smoothing)
            elif type == 'wordKD_criterion':
                teacher = kwargs['teacher']
                init_use_KD = loss.get('init_use_KD', False)
                self.wordKD_criterion = WordKDCriterion(teacher, init_use_KD)

    def forward(self, inputs, labels=None, normalization=1.0, reduce=True, seqs_x=None, y_inp=None, **kwargs):
        loss = 0
        loss_dict = {}
        if self.nmt_criterion is not None:
            assert inputs is not None, u"when compute nmt criterion, input is None"
            assert labels is not None, u"when compute nmt criterion, labels is None"
            nmt_loss = self.nmt_criterion(inputs=inputs, labels=labels, reduce=reduce, normalization=normalization)
            loss_dict[str(self.nmt_criterion)] = nmt_loss
            loss += nmt_loss
        if self.wordKD_criterion is not None:
            if self.wordKD_criterion.use_KD_loss:
                assert seqs_x is not None, u"when compute wordKD criterion, seqs_x is None"
                assert y_inp is not None, u"when compute wordKD criterion, y_inp is None"
                teacher_output = self.wordKD_criterion.get_teacher_output(seqs_x, y_inp)
                wordKD_loss = self.wordKD_criterion(inputs=inputs, labels=teacher_output, reduce=reduce,
                                                    normalization=normalization)
                loss_dict[str(self.wordKD_criterion)] = wordKD_loss
                loss += wordKD_loss
        return loss, loss_dict

    def INFO(self):
        if self.nmt_criterion is not None:
            INFO(self.nmt_criterion)
        if self.wordKD_criterion is not None:
            INFO(self.wordKD_criterion)

    def set_use_KD(self, use_KD):
        if self.wordKD_criterion is not None:
            self.wordKD_criterion.use_KD_loss = use_KD

    def reset_teacher(self, teacher=None):
        if self.wordKD_criterion is not None:
            self.wordKD_criterion.teacher = teacher
