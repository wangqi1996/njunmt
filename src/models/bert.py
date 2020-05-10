# coding=utf-8
from torch import nn

from src.models.transformer import Encoder, Generator, PAD


class Bert(nn.Module):
    """
    Bert model: bidirectional encoder representations for transformers
    """

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None,
            dropout=0.1, tie_input_output_embedding=True, tie_source_target_embedding=False, padding_idx=PAD,
            layer_norm_first=True, positional_embedding="sin", generator_bias=False, ffn_activation="relu", **kwargs):

        super(Bert, self).__init__()

        self.encoder = Encoder(
            n_src_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head,
            padding_idx=padding_idx, layer_norm_first=layer_norm_first, positional_embedding=positional_embedding,
            ffn_activation=ffn_activation)

        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        if tie_input_output_embedding:
            self.generator = Generator(n_words=n_tgt_vocab,
                                       hidden_size=d_word_vec,
                                       shared_weight=self.encoder.embeddings.embeddings.weight,
                                       padding_idx=PAD, add_bias=generator_bias)

        else:
            self.generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=PAD,
                                       add_bias=generator_bias)

    def forward(self, src_seq, tgt_seq, log_probs=True):

        enc_output, enc_mask = self.encoder(src_seq)

        return self.generator(enc_output, log_probs=log_probs)
