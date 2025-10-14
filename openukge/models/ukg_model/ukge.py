import torch
import torch.nn as nn


class UKGE(nn.Module):
    def __init__(self, num_ent=None, num_rel=None, emb_dim=None, reg_scale=None, config=None):
        super(UKGE, self).__init__()
        self.config = config
        self.reg_scale = reg_scale
        self.emb_dim = emb_dim
        self.ent_emb = nn.Embedding(num_ent, emb_dim)
        self.rel_emb = nn.Embedding(num_rel, emb_dim)
        self.w = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.init_emb()

    def init_emb(self):
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

    def tri2emb(self, triples):
        head_emb = self.ent_emb(triples[:, 0])
        relation_emb = self.rel_emb(triples[:, 1])
        tail_emb = self.ent_emb(triples[:, 2])
        return head_emb, relation_emb, tail_emb

    def score_func(self, head_emb, relation_emb, tail_emb):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`h^{\top} \operatorname{diag}(r) t`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.

        Returns:
            score: The score of triples.
        """

        score = head_emb * relation_emb * tail_emb

        score = score.sum(dim=-1)

        """ 1.Bounded rectifier """
        # shape = score.shape
        # tmp_max = torch.max(self.w * score + self.b, torch.zeros(shape, device=self.args.gpu))
        # score = torch.min(tmp_max, torch.ones(shape, device=self.args.gpu))

        """ 2.Logistic function"""
        score = torch.sigmoid(self.w * score + self.b)

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
