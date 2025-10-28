from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompatLayerNorm(nn.Module):
    """Compatibility wrapper for the original LayerNormalization behavior.
    Internally uses nn.LayerNorm for stability and performance but preserves the
    exposed interface.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-3) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original early-return when size(1) == 1 is kept for compatibility
        if x.dim() >= 2 and x.size(1) == 1:
            return x
        return self.layer_norm(x)


class SupportEncoder(nn.Module):
    """Simple feed-forward residual block used to encode support/query vectors."""

    def __init__(self, d_model: int, d_inner: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.norm = CompatLayerNorm(d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.proj1.weight)
        nn.init.xavier_normal_(self.proj2.weight)
        # if self.proj1.bias is not None:
        #     nn.init.zeros_(self.proj1.bias)
        # if self.proj2.bias is not None:
        #     nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.proj1(x))
        out = self.dropout(self.proj2(out))
        return self.norm(out + residual)


class QueryEncoder(nn.Module):
    """RNN-based iterative processor similar to Matching Networks' controller.

    The module avoids creating tensors on specific devices — it uses the
    device of the input tensors.
    """

    def __init__(self, input_dim: int, process_steps: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.process_steps = process_steps
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)

    def forward(self, support: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # support: (support_size, dim) or (1, dim)
        # query: (batch_size, dim)
        assert support.size(1) == query.size(1)

        if self.process_steps == 0:
            return query

        batch_size = query.size(0)
        device = query.device
        h_r = torch.zeros(batch_size, 2 * self.input_dim, device=device)
        c = torch.zeros(batch_size, 2 * self.input_dim, device=device)

        h_out = query
        for _ in range(self.process_steps):
            h_r_, c = self.process(query, (h_r, c))
            h = query + h_r_[:, : self.input_dim]
            attn = F.softmax(torch.matmul(h, support.t()), dim=1)  # (batch, support_size)
            r = torch.matmul(attn, support)
            h_r = torch.cat((h, r), dim=1)
            h_out = h

        return h_out


class MatchNet(nn.Module):
    """RNN Match network used for mean / variance matching.

    Notes:
    - Avoids deprecated Variable usage.
    - Uses device of inputs when creating tensors.
    """

    def __init__(self, input_dim: int, process_steps: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.process_steps = process_steps
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)

    def forward(
        self,
        support_mean: torch.Tensor,
        support_var: Optional[torch.Tensor],
        query_mean: torch.Tensor,
        query_var: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # support_mean: (support_size, dim) or (1, dim)
        # query_mean: (batch, dim)
        assert support_mean.size(1) == query_mean.size(1)

        if self.process_steps == 1:
            return torch.matmul(query_mean, support_mean.t()).squeeze()

        batch_size = query_mean.size(0)
        device = query_mean.device
        h_r = torch.zeros(batch_size, 2 * self.input_dim, device=device)
        c = torch.zeros(batch_size, 2 * self.input_dim, device=device)

        for _ in range(self.process_steps):
            h_r_, c = self.process(query_mean, (h_r, c))
            h = query_mean + h_r_[:, : self.input_dim]
            attn = F.softmax(torch.matmul(h, support_mean.t()), dim=1)
            r = torch.matmul(attn, support_mean)
            h_r = torch.cat((h, r), dim=1)

        matching_scores = torch.matmul(h, support_mean.t()).squeeze()
        return matching_scores


class Matcher(nn.Module):
    """Base matcher with neighbor encoding and common utilities.

    This class contains the main building blocks and helper functions used by
    GMUC. It is kept general so other matching variants can reuse it.
    """

    def __init__(self, num_symbols: int, emb_dim: int, dropout :float, process_steps: int) -> None:
        super().__init__()
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols

        self.symbol_emb = nn.Embedding(num_symbols + 1, emb_dim, padding_idx=self.pad_idx)
        self.symbol_var_emb = nn.Embedding(num_symbols + 1, emb_dim, padding_idx=self.pad_idx)

        # small compatibility layer-norm
        self.layer_norm = CompatLayerNorm(2 * emb_dim)
        self.dropout = nn.Dropout(dropout)

        # aggregation encoder / decoder
        self.set_rnn_encoder = nn.LSTM(2 * emb_dim, 2 * emb_dim, num_layers=1, bidirectional=False)
        self.set_rnn_decoder = nn.LSTM(2 * emb_dim, 2 * emb_dim, num_layers=1, bidirectional=False)

        # attention layers for neighbors and sets
        self.neigh_att_W = nn.Linear(2 * emb_dim, emb_dim)
        self.neigh_att_u = nn.Linear(emb_dim, 1)
        self.neigh_var_att_W = nn.Linear(2 * emb_dim, emb_dim)
        self.neigh_var_att_u = nn.Linear(emb_dim, 1)

        self.set_att_W = nn.Linear(2 * emb_dim, emb_dim)
        self.set_att_u = nn.Linear(emb_dim, 1)

        self.FC_query_g = nn.Linear(2 * emb_dim, 2 * emb_dim)
        self.FC_support_g_encoder = nn.Linear(2 * emb_dim, 2 * emb_dim)

        # initialization
        for m in [
            self.neigh_att_W,
            self.neigh_att_u,
            # self.neigh_var_att_W,
            # self.neigh_var_att_u,
            self.set_att_W,
            self.set_att_u,
            self.FC_query_g,
            self.FC_support_g_encoder,
        ]:
            if hasattr(m, "weight"):
                nn.init.xavier_normal_(m.weight)
                # if hasattr(m, "bias") and m.bias is not None:
                #     nn.init.zeros_(m.bias)


        d_model = emb_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

        self.w = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def neighbor_encoder(self, connections: torch.Tensor, num_neighbors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of neighbors into mean and variance vectors.

        connections: (batch, neighbor_count, 3) -> [relation, entity, confidence]
        num_neighbors: (batch,) number of neighbors for each item (not strictly used but kept for compatibility)
        """
        device = connections.device
        relations = connections[:, :, 0].long()
        entities = connections[:, :, 1].long()
        confidences = connections[:, :, 2]

        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)  # (batch, neighbor, 2*emb_dim)

        rel_var_embeds = self.dropout(self.symbol_var_emb(relations))
        ent_var_embeds = self.dropout(self.symbol_var_emb(entities))
        concat_var_embeds = torch.cat((rel_var_embeds, ent_var_embeds), dim=-1)

        # neighbor attention (works with variable neighbor_size)
        out = torch.tanh(self.neigh_att_W(concat_embeds))  # (batch, neighbor, emb_dim)
        att_w = self.neigh_att_u(out).squeeze(-1)  # (batch, neighbor)
        att_w = F.softmax(att_w, dim=1).unsqueeze(1)  # (batch, 1, neighbor)
        out_mean = torch.bmm(att_w, ent_embeds).squeeze(1)  # (batch, emb_dim)

        out_var = torch.tanh(self.neigh_var_att_W(concat_var_embeds))
        att_var_w = self.neigh_var_att_u(out_var).squeeze(-1)
        att_var_w = F.softmax(att_var_w, dim=1).unsqueeze(1)
        out_var = torch.bmm(att_var_w, ent_var_embeds).squeeze(1)

        return torch.tanh(out_mean), torch.tanh(out_var)

    def score_func(self, support, support_meta, query, query_meta):
        raise NotImplementedError

    def forward(self, support, support_meta, query, query_meta, false, false_meta):
        raise NotImplementedError


class GMUC(Matcher):
    """GMUC: Gaussian Metric Learning for Few-Shot Uncertain KG Completion.

    This class keeps the original logic but uses the refactored utilities above.
    """

    def __init__(self, num_symbols: int, emb_dim: int, dropout: float, process_steps: int) -> None:
        super().__init__(num_symbols, emb_dim, dropout, process_steps)
        self.emb_dim = emb_dim
        d_model = emb_dim * 2

        self.matchnet_mean = MatchNet(d_model, process_steps)
        self.matchnet_var = MatchNet(d_model, process_steps)

        self.gcn_w = nn.Linear(2 * emb_dim, emb_dim)
        self.gcn_w_var = nn.Linear(2 * emb_dim, emb_dim)

        # nn.init.xavier_uniform_(self.gcn_w.weight)
        # nn.init.xavier_uniform_(self.gcn_w_var.weight)
        # if self.gcn_w.bias is not None:
        #     nn.init.zeros_(self.gcn_w.bias)
        # if self.gcn_w_var.bias is not None:
        #     nn.init.zeros_(self.gcn_w_var.bias)

    def score_func(self, support, support_meta, query, query_meta):
        # Unpack meta information
        q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
        s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta

        # ===== 1) neighbor encoding =====
        q_left, q_var_left = self.neighbor_encoder(q_l_conn, q_l_deg)
        q_right, q_var_right = self.neighbor_encoder(q_r_conn, q_r_deg)
        s_left, s_var_left = self.neighbor_encoder(s_l_conn, s_l_deg)
        s_right, s_var_right = self.neighbor_encoder(s_r_conn, s_r_deg)

        # build mean representations
        q_neighbor = torch.cat((q_left, q_right), dim=-1)
        s_neighbor = torch.cat((s_left, s_right), dim=-1)

        # encode support / query via support_encoder
        support_g = self.support_encoder(s_neighbor)
        query_g = self.support_encoder(q_neighbor)

        # position encoder (rnn encoder)
        support_g_0 = support_g.view(-1, 1, 2 * self.emb_dim)  # (support_size, 1, dim)
        support_g_encoder, support_g_state = self.set_rnn_encoder(support_g_0)

        # decoder (run same number of steps as support size)
        last = support_g_encoder[-1].view(1, -1, 2 * self.emb_dim)
        decoder_state = support_g_state
        decoder_outputs = []
        for _ in range(support_g_0.size(0)):
            last, decoder_state = self.set_rnn_decoder(last, decoder_state)
            decoder_outputs.append(last)
        decoder_set = torch.cat(decoder_outputs, dim=0)

        ae_loss = nn.MSELoss()(support_g_0, decoder_set.detach())

        # encoder residual connection similar to original
        support_g_encoder = support_g_0.view(support_g_0.size(0), -1) + support_g_encoder.view(support_g_0.size(0), -1)

        # attention across the encoded support set
        support_g_att = torch.tanh(self.set_att_W(support_g_encoder))
        att_w = self.set_att_u(support_g_att).squeeze(-1)  # (support_size,)
        att_w = F.softmax(att_w, dim=0).unsqueeze(0)  # (1, support_size)
        support_g_encoder = torch.matmul(att_w, support_g_encoder).view(1, -1)

        support_repr = support_g_encoder
        query_repr = query_g

        # variance encodings
        q_var_neighbor = torch.cat((q_var_left, q_var_right), dim=-1)
        s_var_neighbor = torch.cat((s_var_left, s_var_right), dim=-1)
        support_var = torch.mean(s_var_neighbor, dim=0, keepdim=True)

        # ===== 2) matching =====
        matching_scores = self.matchnet_mean(support_repr, None, query_repr, None)
        matching_scores_var = self.matchnet_var(support_var, None, q_var_neighbor, None)
        matching_scores_var = torch.sigmoid(self.w + matching_scores_var + self.b)

        return matching_scores, matching_scores_var, ae_loss

    def forward(self, support, support_meta, query, query_meta, false, false_meta):
        q_scores, q_scores_var, q_ae_loss = self.score_func(support, support_meta, query, query_meta)
        f_scores, f_scores_var, f_ae_loss = self.score_func(support, support_meta, false, false_meta)
        query_confidence = query[:, 2] if query.ndim == 2 and query.size(1) > 2 else None
        return q_scores, q_scores_var, q_ae_loss, f_scores, f_scores_var, f_ae_loss, query_confidence


# End of file
