# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from src.models.base import NMTModel
from src.modules.tensor_utils import FLOAT32_INF
from src.utils.common_utils import Constants
from src.utils.logging import INFO
from .utils import mask_scores, tensor_gather_helper


def beam_search(nmt_model, beam_size, max_steps, src_seqs, alpha=-1.0, sampling_topK=10):
    """

    Args:
        nmt_model (NMTModel):
        beam_size (int):
        max_steps (int):
        src_seqs (torch.Tensor):

    Returns:

    """
    # INFO("sample search")
    batch_size = src_seqs.size(0)

    enc_outputs = nmt_model.encode(src_seqs)
    init_dec_states = nmt_model.init_decoder(enc_outputs, expand_size=beam_size)

    # Prepare for beam searching
    beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
    final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
    beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
    final_word_indices = src_seqs.new(batch_size, beam_size, 1).fill_(Constants.BOS)

    dec_states = init_dec_states

    for t in range(max_steps):
        next_scores, dec_states = nmt_model.decode(final_word_indices.view(batch_size * beam_size, -1), dec_states,
                                                   sample_K=0, seed=0)
        # mask
        next_scores = -next_scores
        next_scores = next_scores.view(batch_size, beam_size, -1)
        for b in range(batch_size):
            if beam_mask[b][0] == 0:
                next_scores[b][0][0] = FLOAT32_INF
        next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask, eos_idx=Constants.EOS)
        next_scores = -next_scores
        batch_size, beam_size, vocab_size = next_scores.shape

        if t == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            next_scores = next_scores[:, ::beam_size, :].contiguous()

        # 从topk中进行采样, next_scores: [bts, beam, topK]
        # 注意，这里next_scores的顺序发生了改变， top_indices指的是当前坐标在原next_scores中的下标，也就是对应的vocab的index
        next_scores, top_indices = next_scores.topk(sampling_topK, dim=-1)
        probs = next_scores.exp()

        # 这个像是从每个beam做一下sample  indices相对于新的next_score
        # sample
        bsz = batch_size
        if t == 0:
            indices_buf = torch.multinomial(
                probs.view(bsz, -1), beam_size, replacement=True,
            ).view(bsz, beam_size)
        else:
            indices_buf = torch.multinomial(
                probs.view(bsz * beam_size, -1),
                1,
                replacement=True,
            ).view(bsz, beam_size)

        if t == 0:
            # expand to beam size
            probs = probs.expand(bsz, beam_size, -1)


        # gather socres , 相对于新的next_score [batch_size, beam_size]
        scores_buf = torch.gather(
            probs, dim=-1, index=indices_buf.unsqueeze(-1)
        )
        scores_buf = scores_buf.log_().view(batch_size, -1)
        # 在词典中的下标
        indices_buf = torch.gather(
            top_indices.expand(batch_size, beam_size, -1),
            dim=2,
            index=indices_buf.unsqueeze(-1)
        ).squeeze(2)

        next_word_ids = indices_buf

        if t != 0:
            beam_scores = scores_buf + beam_scores  # [B, Bm] + [B, Bm]

        # If next_word_ids is EOS, beam_mask_ should be 0.0
        beam_mask_ = 1.0 - next_word_ids.eq(Constants.EOS).float()
        next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                   Constants.PAD)  # If last step a EOS is already generated, we replace the last token as PAD
        beam_mask = beam_mask * beam_mask_

        # # If an EOS or PAD is encountered, set the beam mask to 0.0
        final_lengths += beam_mask

        final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

        if beam_mask.eq(0.0).all():
            break

    # Length penalty
    if alpha > 0.0:
        scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
    else:
        scores = beam_scores / final_lengths

    _, reranked_ids = torch.sort(scores, dim=-1, descending=True)

    return tensor_gather_helper(gather_indices=reranked_ids,
                                gather_from=final_word_indices[:, :, 1:].contiguous(),
                                batch_size=batch_size,
                                beam_size=beam_size,
                                gather_shape=[batch_size * beam_size, -1])
