from torch import nn

import src.utils.init as my_init
from src.modules.embeddings import Embeddings


class LM(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, vocab_size, embedding_size=300, hidden_size=512, num_layers=2, dropout=0.3, shared_weight=True,
            **kwargs):
        super().__init__()
        self.embedding = Embeddings(num_embeddings=embedding_size,
                                    embedding_dim=vocab_size,
                                    dropout=dropout)

        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True
        )
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embedding_size)
        )

        # 投影层
        self.proj = nn.Linear(embedding_size, vocab_size, bias=False)
        if shared_weight:
            self.proj.weight = self.embedding.weight
        else:
            my_init.default_init(self.proj.weight)

    def forward(self, seq):
        input = self.embedding(seq)
        hidden, _ = self.rnn(input)
        output = self.output(hidden)
        logit = self.proj(output)
        return logit
