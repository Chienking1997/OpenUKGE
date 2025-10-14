import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension, device='cpu'):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.device = device

        self.w_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
        self._init_emb()

    def _init_emb(self):
        init_range = 0.5 / self.emb_dimension
        self.w_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.zero_()

    def forward(self, pos_w, pos_v, neg_v):
        pos_w = torch.tensor(pos_w, dtype=torch.long, device=self.device)
        pos_v = torch.tensor(pos_v, dtype=torch.long, device=self.device)
        neg_v = torch.tensor(neg_v, dtype=torch.long, device=self.device)

        emb_w = self.w_embeddings(pos_w)  # Size: [mini_batch_size, emb_dimension]
        emb_v = self.v_embeddings(pos_v)  # Size: [mini_batch_size, emb_dimension]
        neg_emb_v = self.v_embeddings(neg_v)  # Size: [neg_count, mini_batch_size, emb_dimension]

        score = torch.mul(emb_w, emb_v).sum(dim=1)
        score = F.logsigmoid(score)

        neg_score = torch.bmm(neg_emb_v, emb_v.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        # Loss: L = log sigmoid (Xw.T * θv) + ∑neg(v) [log sigmoid (-Xw.T * θneg(v))]
        loss = -1 * (score.sum() + neg_score.sum())

        return loss

    def save_embedding(self, id2word_dict, file_name):
        embedding = self.w_embeddings.weight.data
        with open(file_name, 'w', encoding='utf-8') as file_output:
            file_output.write(f'{self.emb_size} {self.emb_dimension}\n')
            for id, word in id2word_dict.items():
                e = embedding[id].tolist()
                e_str = ' '.join(map(str, e))
                file_output.write(f'{word} {e_str}\n')

    def vec_embedding(self):
        return self.w_embeddings.weight.data
