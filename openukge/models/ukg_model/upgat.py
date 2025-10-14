import torch
import torch.nn as nn


class UPGAT(nn.Module):
    def __init__(self, num_ent=None, num_rel=None, emb_dim=None, reg_scale=None, config=None):
        super(UPGAT, self).__init__()
        self.config = config
        self.reg_scale = reg_scale
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.ent_emb = nn.Embedding(num_ent, emb_dim)
        self.rel_emb = nn.Embedding(num_rel, emb_dim)
        self.w = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.init_emb()

    def init_emb(self):
        """Initialize the model and entity and relation embeddings.

        Args:
            ent_emb_init: Entity embedding, shape:[num_ent, emb_dim].
            rel_emb_init: Relation embedding, shape:[num_rel, emb_dim].
            ent_emb: The final embedding used in the model.
            rel_emb: The final embedding used in the model.
            W_1: Weight when calculate attention scores.
            W_a: Weight when calculate attention scores.
            g_0: Embedding for the special self-loop relation of the attention baseline.
            W_E: Weight when update entity embedding.
            W_R: Weight when update relation embedding.
        """
        self.ent_emb_init = nn.Parameter(torch.randn(self.num_ent, self.emb_dim))
        self.rel_emb_init = nn.Parameter(torch.randn(self.num_rel, self.emb_dim))
        self.ent_emb = nn.Parameter(torch.randn(self.num_ent, self.emb_dim))
        self.rel_emb = nn.Parameter(torch.randn(self.num_rel, self.emb_dim))

        self.W_1 = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim))
        self.W_a = nn.Parameter(torch.zeros(1, self.emb_dim))
        self.g_0 = nn.Parameter(torch.zeros(1, self.emb_dim))
        self.W_E = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim))
        self.W_R = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim))
        nn.init.xavier_uniform_(self.g_0.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_a.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_E.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_R.data, gain=1.414)

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

    def forward_GAT(self, triples, adj_matrix):
        """The functions used in the training phase for updating embedding of triples.

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            adj_matrix: The adjacency matrix of the triples.
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            head_emb: Head entity embedding.
            relation_emb: Relation embedding.
            tail_emb: Tail entity embedding.
        """
        node_list = adj_matrix[0].to(self.args.gpu)  # ent
        edge_list = adj_matrix[1].to(self.args.gpu)  # rel
        self.ent_emb_init.data = F.normalize(self.ent_emb_init.data, p=2, dim=1).detach()
        h_i = self.ent_emb_init[node_list[0, :], :]
        h_j = self.ent_emb_init[node_list[1, :], :]
        g_k = self.rel_emb_init[edge_list]

        c_ijk = self.W_1.mm((h_j * g_k).t())
        a_ijk = -nn.LeakyReLU(0.2)(self.W_a.mm(h_i.t() * torch.tanh(c_ijk)).squeeze())

        # get attention scores
        c_baseline = self.W_1.mm((h_i * self.g_0).t())
        a_baseline = -nn.LeakyReLU(0.2)(self.W_a.mm(h_i.t() * torch.tanh(c_baseline)).squeeze())

        a_baseline_exp = torch.exp(a_baseline).unsqueeze(1)
        a_ijk_exp = torch.exp(a_ijk).unsqueeze(1)
        assert not torch.isnan(a_baseline_exp).any()
        assert not torch.isnan(a_ijk_exp).any()

        a_ijk_exp_sum = self.special_spmm_final(node_list, a_ijk_exp, self.args.num_ent, a_ijk_exp.shape[0], 1)
        a_baseline_sorted = torch.zeros_like(a_ijk_exp_sum).to(self.args.gpu)
        a_baseline_sorted[node_list[0]] = a_baseline_exp
        denominator = a_baseline_sorted + a_ijk_exp_sum
        denominator[denominator == 0.0] = 1e-12

        a_ijk_exp = a_ijk_exp.squeeze(1)
        a_ijk_exp = nn.Dropout(0.3)(a_ijk_exp)
        a_baseline_exp = a_baseline_exp.squeeze(1)
        a_baseline_exp = nn.Dropout(0.3)(a_baseline_exp)

        ac_ijk = (a_ijk_exp * c_ijk).t()
        ac_baseline = (a_baseline_exp * c_baseline).t()

        a_ijk_new = self.special_spmm_final(node_list, ac_ijk, self.args.num_ent, ac_ijk.shape[0],
                                            self.args.emb_dim)
        a_baseline_new = torch.zeros_like(a_ijk_new).to(self.args.gpu)
        a_baseline_new[node_list[0]] = ac_baseline

        assert not torch.isnan(a_baseline_new).any()
        assert not torch.isnan(a_ijk_new).any()

        h_prime = F.elu((a_baseline_new + a_ijk_new).div(denominator))
        assert not torch.isnan(h_prime).any()

        # update embeddings
        ent_emb_new = self.ent_emb_init.mm(self.W_E) + h_prime
        rel_emb_new = self.rel_emb_init.mm(self.W_R)
        ent_emb_new = F.normalize(ent_emb_new, p=2, dim=1)

        self.ent_emb.data = ent_emb_new.data
        self.rel_emb.data = rel_emb_new.data

        if mode == "single":
            head_emb = ent_emb_new[triples[:, 0]].unsqueeze(1)  # [bs, 1, dim]
            relation_emb = rel_emb_new[triples[:, 1]].unsqueeze(1)  # [bs, 1, dim]
            tail_emb = ent_emb_new[triples[:, 2]].unsqueeze(1)  # [bs, 1, dim]
        elif mode == "head-batch":
            head_emb = ent_emb_new[negs]  # [bs, num_neg, dim]
            relation_emb = rel_emb_new[triples[:, 1]].unsqueeze(1)  # [bs, 1, dim]
            tail_emb = ent_emb_new[triples[:, 2]].unsqueeze(1)  # [bs, 1, dim]
        elif mode == "tail-batch":
            head_emb = ent_emb_new[triples[:, 0]].unsqueeze(1)  # [bs, 1, dim]
            relation_emb = rel_emb_new[triples[:, 1]].unsqueeze(1)  # [bs, 1, dim]
            tail_emb = ent_emb_new[negs]

        return head_emb, relation_emb, tail_emb

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

class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """
    Special function for only sparse region backpropataion layer, similar to https://arxiv.org/abs/1710.10903
    """
    @staticmethod
    def forward(ctx, node, node_w, N, E, out_features):

        a = torch.sparse_coo_tensor(
            node, node_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            grad_values = grad_output[edge_sources]

        return None, grad_values, None, None, None

class SpecialSpmmFinal(nn.Module):
    """
    Special spmm final layer, similar to https://arxiv.org/abs/1710.10903.
    """
    def forward(self, node, node_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(node, node_w, N, E, out_features)