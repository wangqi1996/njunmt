import torch
import torch.nn as nn

from src.data.vocabulary import PAD


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

    def forward(self, inputs, labels, normalization=1.0, reduce=True, **kwargs):
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


class WordKDCriterion(Criterion):
    """
    use word-level knowledge distillation loss function
    """

    def __init__(self, teacher=None):
        super().__init__()
        self.teacher = teacher
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.use_KD_loss = False

    def reset_teacher(self, teacher=None):
        self.teacher = teacher

    def _compute_loss(self, inputs, labels, teacher_output=None, **kwargs):
        batch_size = labels.size(0)
        loss = self.criterion(inputs, teacher_output).view((batch_size, -1)).sum(-1)
        return loss

    def get_teacher_output(self, input, y_inp):
        assert self.teacher is not None, u"must have a teacher when calculate KD"
        output = self.teacher(input, y_inp, log_probs=False)
        return output

    def use_KD(self, best_bleu_step, last_bleu_step):
        if best_bleu_step == last_bleu_step:
            self.use_KD_loss = False
        else:
            self.use_KD_loss = True

class CombinationCriterion(Criterion):
    """
    use multi loss function for train
    """
    def __init__(self, loss_config):
        """
        """
        for