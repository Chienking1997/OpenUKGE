import torch
import torch.nn as nn


class UKGsE(nn.Module):
    def __init__(self, word_embedding=None,
                 word_id_dict=None,
                 num_ent=None,
                 num_rel=None,
                 emb_dim=None,
                 reg_scale=None,
                 config=None):
        super(UKGsE, self).__init__()
        self.config = config
        self.reg_scale = reg_scale
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.ent_emb = nn.Embedding(num_ent, emb_dim)
        self.rel_emb = nn.Embedding(num_rel, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size=emb_dim, batch_first=True)
        self.fc = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()
        if word_embedding is not None:
            self.init_emb(word_embedding, word_id_dict)
        else:
            self.xavier_init_emb()

    def init_emb(self, word_embedding, word_id_dict):
        for i in range(self.num_ent):
            self.ent_emb.weight.data[i] = word_embedding[word_id_dict[i]]
        for i in range(self.num_rel):
            self.ent_emb.weight.data[i] = word_embedding[word_id_dict['r' + str(i)]]

    def xavier_init_emb(self):
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

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

        Returns:
            score: The score of triples.
        """

        x_triple = torch.stack((head_emb, relation_emb, tail_emb), dim=1)  # [bs, 3, dim]

        lstm_output, _ = self.lstm(x_triple)
        output = self.fc(lstm_output[:, -1, :]).squeeze()
        score = self.sigmoid(output)

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

        return score

    def get_tail_score(self, head_id, relation_id):
        head_emb = self.ent_emb(head_id.repeat(self.num_ent))
        relation_emb = self.rel_emb(relation_id.repeat(self.num_ent))
        score = self.score_func(head_emb, relation_emb,
                                self.ent_emb.weight.data)
        return score

    def get_head_score(self, tail_id, relation_id):
        tail_emb = self.ent_emb(tail_id.repeat(self.num_ent))
        relation_emb = self.rel_emb(relation_id.repeat(self.num_ent))
        score = self.score_func(self.ent_emb.weight.data, relation_emb, tail_emb)
        return score

    def get_hrt_score(self, head_id, relation_id, tail_id):
        head_emb = self.ent_emb(head_id.repeat(len(tail_id)))
        relation_emb = self.rel_emb(relation_id.repeat(len(tail_id)))
        tail_emb = self.ent_emb(tail_id)
        score = self.score_func(head_emb, relation_emb, tail_emb)
        return score

    def get_trh_score(self, head_id, relation_id, tail_id):
        head_emb = self.ent_emb(head_id)
        relation_emb = self.rel_emb(relation_id.repeat(len(head_id)))
        tail_emb = self.ent_emb(tail_id.repeat(len(head_id)))
        score = self.score_func(head_emb, relation_emb, tail_emb)
        return score
