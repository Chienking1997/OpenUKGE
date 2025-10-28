import torch
import torch.nn as nn
from torch import Tensor


class Word2VecSkipGramModel(nn.Module):
    """Word2Vec model with negative sampling, supporting confidence weights on positive samples only."""

    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_emb.weight)
        nn.init.xavier_uniform_(self.output_emb.weight)

    def forward_skipgram(self, center: Tensor, context: Tensor, negatives: Tensor, weights: Tensor = None):
        emb_c = self.input_emb(center)
        emb_o = self.output_emb(context)
        pos_score = torch.sum(emb_c * emb_o, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        neg_emb = self.output_emb(negatives)
        neg_score = torch.bmm(neg_emb, emb_c.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)

        # 只对正样本加权
        if weights is not None:
            loss = -(pos_loss * weights + neg_loss)
        else:
            loss = -(pos_loss + neg_loss)
        return loss.mean()


