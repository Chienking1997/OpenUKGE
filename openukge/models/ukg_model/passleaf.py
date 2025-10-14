import torch
import torch.nn as nn


class PASSLEAF(nn.Module):
    """`A Pool-bAsed Semi-Supervised LEArning Framework for Uncertain Knowledge Graph Embedding`_ (PASSLEAF).

    Attributes:

        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim].
        w: Weight when calculate confidence scores
        b: Bias when calculate confidence scores
        Translating Embeddings for Modeling Multi-relational Data: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
    """

    def __init__(self, num_ent=None, num_rel=None, emb_dim=None,
                 reg_scale=None, margin=None, score_function=None, config=None):
        super(PASSLEAF, self).__init__()
        self.config = config
        self.score_function = score_function
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.reg_scale = reg_scale
        self.margin = margin
        self.ent_emb = None
        self.rel_emb = None
        self.w = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.embedding_range = None
        self.epsilon = None

        self.init_emb()

    def init_emb(self):
        """
            Initialize the entity and relation embeddings in the form of a uniform distribution.
        """
        model = self.score_function
        if model == 'DistMult':
            "`Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_ (DistMult)"
            self.ent_emb = nn.Embedding(self.num_ent, self.emb_dim)
            self.rel_emb = nn.Embedding(self.num_rel, self.emb_dim)
            nn.init.xavier_uniform_(self.ent_emb.weight.data)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)
        elif model == 'ComplEx':
            "`Complex Embeddings for Simple Link Prediction`_ (ComplEx)"
            self.ent_emb = nn.Embedding(self.num_ent, self.emb_dim * 2)
            self.rel_emb = nn.Embedding(self.num_rel, self.emb_dim * 2)
            nn.init.xavier_uniform_(self.ent_emb.weight.data)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)
        else:
            "`RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_ (RotatE)"
            """Initialize the entity and relation embeddings in the form of a uniform distribution."""
            # print(self.gama)
            """Initialize the entity and relation embeddings in the form of a uniform distribution."""
            self.epsilon = 2.0
            self.margin = nn.Parameter(torch.Tensor([self.margin]), requires_grad=False)
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin.item() + self.epsilon) / self.emb_dim]),
                requires_grad=False
            )
            self.ent_emb = nn.Embedding(self.num_ent, self.emb_dim * 2)
            self.rel_emb = nn.Embedding(self.num_rel, self.emb_dim)
            nn.init.xavier_uniform_(self.ent_emb.weight.data)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)

    def tri2emb(self, triples):
        head_emb = self.ent_emb(triples[:, 0])
        relation_emb = self.rel_emb(triples[:, 1])
        tail_emb = self.ent_emb(triples[:, 2])
        return head_emb, relation_emb, tail_emb

    def score_func(self, head_emb, relation_emb, tail_emb):
        """Calculating the score of triples.

        The formula for calculating the score of DistMult is :math:`h^{T} \operatorname{diag}(r) t`.

        The formula for calculating the score of ComplEx is :math:`Re(h^{T} \operatorname{diag}(r) \overline{t})`.

        The formula for calculating the score of RotatE is :math:`\gamma - \|h \circ r - t\|`

        Args:


        Returns:
            score: The score of triples.
        """

        score = None
        model = self.score_function  # use different score function according to args.passleaf_score_function

        if model == 'DistMult':
            score = head_emb * relation_emb * tail_emb
            score = score.sum(dim=-1)

            """ 1.Bounded rectifier """
            # shape = score.shape
            # tmp_max = torch.max(self.w * score + self.b, torch.zeros(shape, device=self.args.gpu))
            # score = torch.min(tmp_max, torch.ones(shape, device=self.args.gpu))

            """ 2.Logistic function"""
            score = torch.sigmoid(self.w * score + self.b)  # use UKGE_logi in PASSLEAF

        if model == 'ComplEx':
            re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
            re_relation, im_relation = torch.chunk(relation_emb, 2, dim=-1)
            re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

            score = torch.sum(
                re_head * re_tail * re_relation
                + im_head * im_tail * re_relation
                + re_head * im_tail * im_relation
                - im_head * re_tail * im_relation,
                -1
            )
            score = torch.sigmoid(self.w * score + self.b)

        if model == 'RotatE':
            re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
            re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

            # Make phases of relations uniformly distributed in [-pi, pi]

            pi = 3.14159265358979323846
            phase_relation = relation_emb / (self.embedding_range.item() / pi)

            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)

            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

            score = torch.stack([re_score, im_score], dim=0)
            score = score.norm(dim=0)
            score = self.margin.item() - score.sum(dim=-1)
            score = torch.sigmoid(self.w * score + self.b)

        return score

    def forward(self, triples):
        """The functions used in the training phase

        Args:


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
