import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOWModel(nn.Module):
    def __init__(self, emb_size, emb_dimension, device):
        super(CBOWModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.device = device
        self.u_embeddings = nn.Embedding(self.emb_size, self.emb_dimension)  # Define input embeddings
        self.w_embeddings = nn.Embedding(self.emb_size, self.emb_dimension)  # Define output embeddings
        self._init_embedding()  # Initialize embeddings

    def _init_embedding(self):
        int_range = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-int_range, int_range)
        self.w_embeddings.weight.data.zero_()

    # Forward pass
    def forward(self, pos_u, pos_w, neg_w):
        pos_u_emb = []
        for per_Xw in pos_u:
            per_u_emb = self.u_embeddings(
                torch.LongTensor(per_Xw).to(self.device))  # Embedding lookup for context words
            per_u_sum = torch.sum(per_u_emb, dim=0)  # Sum over embeddings
            pos_u_emb.append(per_u_sum)

        pos_u_emb = torch.stack(pos_u_emb)  # Stack into tensor of size [batch_size, emb_dimension]
        pos_w_emb = self.w_embeddings(torch.LongTensor(pos_w).to(self.device))  # [batch_size, emb_dimension]
        neg_w_emb = self.w_embeddings(
            torch.LongTensor(neg_w).to(self.device))  # [neg_samples, batch_size, emb_dimension]

        # Compute positive score
        score_1 = torch.mul(pos_u_emb, pos_w_emb).sum(dim=1)  # Dot product
        score_2 = F.logsigmoid(score_1)  # log sigmoid

        # Compute negative score
        neg_score_1 = torch.bmm(neg_w_emb, pos_u_emb.unsqueeze(2)).squeeze(2)  # Batch matrix multiplication
        neg_score_2 = F.logsigmoid(-neg_score_1).sum(dim=1)  # Log sigmoid and sum

        # Loss function
        loss = -torch.sum(score_2 + neg_score_2)
        return loss

    # Save embeddings
    def save_embedding(self, id2word_dict, file_name):
        embedding = self.u_embeddings.weight.data
        with open(file_name, 'w', encoding='utf-8') as file_output:
            file_output.write(f'{self.emb_size} {self.emb_dimension}\n')
            for id, word in id2word_dict.items():
                e = embedding[id].tolist()
                e_str = ' '.join(map(str, e))
                file_output.write(f'{word} {e_str}\n')

    def vec_embedding(self):
        return self.u_embeddings.weight.data
