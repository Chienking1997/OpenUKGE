import torch
import torch.nn as nn


class GtransE(nn.Module):
    def __init__(self, num_ent=None, num_rel=None, emb_dim=None, margin= None, reg_scale=None, config=None):
        super(GtransE, self).__init__()
        self.config = config
        self.reg_scale = reg_scale
        self.margin = margin
        self.emb_dim = emb_dim
        self.ent_emb = nn.Embedding(num_ent, emb_dim)
        self.rel_emb = nn.Embedding(num_rel, emb_dim)
        self.norm = 1
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.margin]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.emb_dim]),
            requires_grad=False
        )

        self.init_emb2()

    def init_emb(self):
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

    def init_emb2(self):
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def tri2emb(self, triples):
        head_emb = self.ent_emb(triples[:, 0])
        relation_emb = self.rel_emb(triples[:, 1])
        tail_emb = self.ent_emb(triples[:, 2])
        return head_emb, relation_emb, tail_emb

    def score_func(self, head_emb, relation_emb, tail_emb):
        """Calculating the score of triples.

                Args:
                    head_emb: The head entity embedding.
                    relation_emb: The relation embedding.
                    tail_emb: The tail entity embedding.
                    mode: Choose head-predict or tail-predict.

                Returns:
                    score: The score of triples.
                """

        score = (head_emb + relation_emb) - tail_emb
        score = - torch.norm(score, p=self.norm, dim=-1)

        return score

    def forward(self, triples):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t, c), shape:[batch_size, 3].

        Returns:
            score: The score of triples.
        """
        head_emb, relation_emb, tail_emb = self.tri2emb(triples)
        score = self.score_func(head_emb, relation_emb, tail_emb)

        return score

    def regularization(self, triples):
        # Only pos_sample Weight Decay
        head_emb, relation_emb, tail_emb = self.tri2emb(triples)
        return self.reg_scale * (torch.mean(head_emb ** 2) +
                                 torch.mean(relation_emb ** 2) +
                                 torch.mean(tail_emb ** 2))

    def get_tail_score(self, head_id, relation_id):
        head_emb = self.ent_emb(head_id)
        relation_emb = self.rel_emb(relation_id)
        score = self.score_func(head_emb, relation_emb, self.ent_emb.weight.data)
        return score

    def get_head_score(self, tail_id, relation_id):
        tail_emb = self.ent_emb(tail_id)
        relation_emb = self.rel_emb(relation_id)
        score = self.score_func(self.ent_emb.weight.data, relation_emb, tail_emb)
        return score

    def get_hrt_score(self, head_id, relation_id, tail_id):
        head_emb = self.ent_emb(head_id)
        relation_emb = self.rel_emb(relation_id)
        tail_emb = self.ent_emb(tail_id)
        score = self.score_func(head_emb, relation_emb, tail_emb)
        return score
