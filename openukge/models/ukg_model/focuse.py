import torch
import torch.nn as nn
import torch.nn.functional as F


class FocusE(nn.Module):
    def __init__(self, num_ent=None, num_rel=None, emb_dim=None,
                 base_model=None, margin=None, reg_scale1=None,
                 reg_scale2=None, config=None):
        super(FocusE, self).__init__()
        self.embedding_range = None
        self.epsilon = None
        self.config = config
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.reg_scale1 = reg_scale1
        self.reg_scale2 = reg_scale2
        self.emb_dim = emb_dim
        self.margin = margin
        self.base_model = base_model
        self.ent_emb = nn.Embedding(self.num_ent, self.emb_dim)
        self.rel_emb = nn.Embedding(self.num_rel, self.emb_dim)

        # β 表示结构影响系数（随 epoch 变化）
        self.beta = 1.0

        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution.

        """
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.margin]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.emb_dim]),
            requires_grad=False
        )

        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        # nn.init.xavier_uniform_(self.ent_emb.weight.data)
        # nn.init.xavier_uniform_(self.rel_emb.weight.data)

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

        if self.base_model == "DistMult":
            score = (head_emb * tail_emb * relation_emb).sum(dim=-1)

        elif self.base_model == "ComplEX":
            re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
            re_relation, im_relation = torch.chunk(relation_emb, 2, dim=-1)
            re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

            score = torch.sum(re_head * re_tail * re_relation +
                              im_head * im_tail * re_relation +
                              re_head * im_tail * im_relation -
                              im_head * re_tail * im_relation, -1
                              )
        else:  # TransE
            score = (head_emb + relation_emb) - tail_emb
            score = self.margin.item() - torch.norm(score, p=1, dim=-1)


        return score

    def forward(self, triples):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].

        Returns:
            score: The score of triples.
        """
        head_emb, relation_emb, tail_emb = self.tri2emb(triples)
        score = self.score_func(head_emb, relation_emb, tail_emb)
        score = F.softplus(score)
        return score

    def forward_weighted(self, triples, pro, pro_expand=None):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            pro: The pro c, shape:[1].

        Returns:
            score: The score of triples.
        """
        head_emb, relation_emb, tail_emb = self.tri2emb(triples)
        score = self.score_func(head_emb, relation_emb, tail_emb)
        score = F.softplus(score)

        if pro_expand is not None:
            score = score.view(pro_expand.shape[0], -1)
            alpha = self.beta + pro_expand * (1 - self.beta)
        else:
            alpha = self.beta + (torch.ones(pro.shape).to(pro.device) - pro) * (1 - self.beta)

        score = alpha * score
        return score

    def regularization(self, triples):
        # Only pos_sample Weight Decay
        head_emb, relation_emb, tail_emb = self.tri2emb(triples)
        return self.reg_scale1 * (torch.mean(head_emb ** 2) +
                                  torch.mean(relation_emb ** 2) +
                                  torch.mean(tail_emb ** 2))

    def regularization2(self):
        regularization = self.reg_scale2 * (
                self.ent_emb.weight.norm(p=3) ** 3 +
                self.rel_emb.weight.norm(p=3) ** 3
        )
        return regularization

    def get_tail_score(self, head_id, relation_id):
        head_emb = self.ent_emb(head_id)
        relation_emb = self.rel_emb(relation_id)
        score = self.score_func(head_emb, relation_emb, self.ent_emb.weight.data)
        score = F.softplus(score)
        return score

    def get_head_score(self, tail_id, relation_id):
        tail_emb = self.ent_emb(tail_id)
        relation_emb = self.rel_emb(relation_id)
        score = self.score_func(self.ent_emb.weight.data, relation_emb, tail_emb)
        score = F.softplus(score)
        return score

    def get_hrt_score(self, head_id, relation_id, tail_id):
        head_emb = self.ent_emb(head_id)
        relation_emb = self.rel_emb(relation_id)
        tail_emb = self.ent_emb(tail_id)
        score = self.score_func(head_emb, relation_emb, tail_emb)
        score = F.softplus(score)
        return score

    def adjust_parameters(self, current_epoch, max_epochs):
        """To adjust the structural influence beta

        Args:
            current_epoch: Which epoch is currently being trained
        """
        self.beta = max(0.0, 1.0 - (current_epoch / max_epochs))
