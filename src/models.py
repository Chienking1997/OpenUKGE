from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import codecs
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from src import param

class U_ANALOGY(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(U_ANALOGY, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self.function = param.function
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim)
        self.ent_embeddings_real = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim // 2)
        self.ent_embeddings_img = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim // 2)
        self.rel_embeddings_real = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim // 2)
        self.rel_embeddings_img = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim // 2)
        nn.init.xavier_uniform_(self.ent_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_real.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_img.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_real.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_img.weight)
        self.liner = torch.nn.Linear(1, 1).cuda()
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)



    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()
        # s_h = torch.tensor(s_h, dtype=torch.int64).cuda()
        # s_r = torch.tensor(s_r, dtype=torch.int64).cuda()
        # s_t = torch.tensor(s_t, dtype=torch.int64).cuda()
        # s_w = torch.tensor(s_w, dtype=torch.float32).cuda()
        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        # psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss #+ psl_loss
        return self._A_loss

    def embed_complex(self, h, r, t):
        h_emb_real = self.ent_embeddings_real(h)
        h_emb_img = self.ent_embeddings_img(h)

        r_emb_real = self.rel_embeddings_real(r)
        r_emb_img = self.rel_embeddings_img(r)

        t_emb_real = self.ent_embeddings_real(t)
        t_emb_img = self.ent_embeddings_img(t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img
    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h_emb = self.ent_embedding(h)
        r_emb = self.rel_embedding(r)
        t_emb = self.ent_embedding(t)

        return h_emb, r_emb, t_emb

    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        h_e, r_e, t_e = self.embed(h, r, t)
        n_hn_e, n_rel_hn_e, n_t_e = self.embed(n_hn, n_rel_hn, n_t)
        n_h_e, n_rel_tn_e, n_tn_e = self.embed(n_h, n_rel_tn, n_tn)
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed_complex(h, r, t)
        n_hn_e_real, n_hn_e_img, n_rel_hn_e_real, n_rel_hn_e_img, n_t_e_real, n_t_e_img = self.embed_complex(n_hn, n_rel_hn, n_t)
        n_h_e_real, n_h_e_img, n_rel_tn_e_real, n_rel_tn_e_img, n_tn_e_real, n_tn_e_img = self.embed_complex(n_h, n_rel_tn, n_tn)

        htr = torch.unsqueeze(torch.sum(r_e*(h_e*t_e), dim=1), dim=-1)
        htr_complex = torch.unsqueeze(torch.sum(h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real +
                      h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img, dim=1), dim=-1)
        f_prob_h = self.liner(htr + htr_complex)
        f_prob_hn = self.liner(torch.unsqueeze(torch.sum(n_hn_e_real * n_t_e_real * n_rel_hn_e_real +
                                             n_hn_e_img * n_t_e_img * n_rel_hn_e_real +
                                             n_hn_e_real * n_t_e_img * n_rel_hn_e_img -
                                             n_hn_e_img * n_t_e_real * n_rel_hn_e_img, dim=2), dim=-1) +
                   torch.unsqueeze(torch.sum(n_rel_hn_e * (n_hn_e * n_t_e), dim=2), dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(torch.sum(n_h_e_real * n_tn_e_real * n_rel_tn_e_real +
                                             n_h_e_img * n_tn_e_img * n_rel_tn_e_real +
                                             n_h_e_real * n_tn_e_img * n_rel_tn_e_img -
                                             n_h_e_img * n_tn_e_real * n_rel_tn_e_img, dim=2), dim=-1) +
                   torch.unsqueeze(torch.sum(n_rel_tn_e * (n_h_e * n_tn_e), dim=2), dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)

        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h-w)

        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)

        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)

        this_loss = (torch.sum(((f_score_tn+f_score_hn)/2.0)*self._p_neg+f_score_h)) / self.batch_size
        regularizer = ((torch.sum(torch.square(h_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(h_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(h_e_img)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r_e_img)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t_e_img)) / 2.0) / self.batch_size)
        main_loss = this_loss + self.reg_scale * regularizer

        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_e_h, s_e_r, s_e_t = self.embed(s_h, s_r, s_t)
        sh_real, sh_img, sr_real, sr_img, st_real, st_img = self.embed_complex(s_h, s_r, s_t)
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_e_r*(s_e_h*s_e_t), dim=1), dim=-1)+
                                            torch.unsqueeze(torch.sum(sh_real * st_real * sr_real +
                                                                      sh_img * st_img * sr_real +
                                                                      sh_real * st_img * sr_img -
                                                                      sh_img * st_real * sr_img, dim=1), dim=-1)), dim=-1)
        # prior_psl0 = torch.tensor(self._prior_psl)
        psl_error_each = torch.square(s_w - psl_prob)
        psl_mse = torch.mean(psl_error_each)
        # psl_loss = (psl_mse * self._p_psl)
        return psl_mse
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

class U_ComplEx(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(U_ComplEx, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = param.function
        # self.ent_embedding = torch.nn.Embedding(num_embeddings=self.num_cons,
        #                                         embedding_dim=self.dim).cuda()
        # self.rel_embedding = torch.nn.Embedding(num_embeddings=self.num_rels,
        #                                         embedding_dim=self.dim).cuda()
        self.ent_embeddings_real = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim)
        self.ent_embeddings_img = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim)
        self.rel_embeddings_real = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim)
        self.rel_embeddings_img = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim)
        nn.init.xavier_uniform_(self.ent_embeddings_real.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_img.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_real.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_img.weight)
        self.liner = torch.nn.Linear(1, 1).cuda()
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)



    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()
        # s_h = torch.tensor(s_h, dtype=torch.int64).cuda()
        # s_r = torch.tensor(s_r, dtype=torch.int64).cuda()
        # s_t = torch.tensor(s_t, dtype=torch.int64).cuda()
        # s_w = torch.tensor(s_w, dtype=torch.float32).cuda()
        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        # psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss #+ psl_loss
        return self._A_loss

    def embed(self, h, r, t):
        h_emb_real = self.ent_embeddings_real(h)
        h_emb_img = self.ent_embeddings_img(h)

        r_emb_real = self.rel_embeddings_real(r)
        r_emb_img = self.rel_embeddings_img(r)

        t_emb_real = self.ent_embeddings_real(t)
        t_emb_img = self.ent_embeddings_img(t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)
        n_hn_e_real, n_hn_e_img, n_rel_hn_e_real, n_rel_hn_e_img, n_t_e_real, n_t_e_img = self.embed(n_hn, n_rel_hn, n_t)
        n_h_e_real, n_h_e_img, n_rel_tn_e_real, n_rel_tn_e_img, n_tn_e_real, n_tn_e_img = self.embed(n_h, n_rel_tn, n_tn)

        htr = torch.unsqueeze(torch.sum(h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real +
                      h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img, dim=1), dim=-1)
        f_prob = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze(torch.sum(n_hn_e_real * n_t_e_real * n_rel_hn_e_real +
                                             n_hn_e_img * n_t_e_img * n_rel_hn_e_real +
                                             n_hn_e_real * n_t_e_img * n_rel_hn_e_img -
                                             n_hn_e_img * n_t_e_real * n_rel_hn_e_img, dim=2), dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(torch.sum(n_h_e_real * n_tn_e_real * n_rel_tn_e_real +
                                             n_h_e_img * n_tn_e_img * n_rel_tn_e_real +
                                             n_h_e_real * n_tn_e_img * n_rel_tn_e_img -
                                             n_h_e_img * n_tn_e_real * n_rel_tn_e_img, dim=2), dim=-1))
        if self.function == 'logi':
            f_prob = torch.sigmoid(f_prob)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)

        f_prob = torch.squeeze(f_prob, dim=-1)
        f_score = torch.square(f_prob-w)

        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)

        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)

        this_loss = (torch.sum(((f_score_tn+f_score_hn)/2.0)*self._p_neg+f_score)) / self.batch_size
        regularizer = ((torch.sum(torch.square(h_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(h_e_img)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r_e_img)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t_e_img)) / 2.0) / self.batch_size)
        main_loss = this_loss + self.reg_scale * regularizer

        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.ent_embedding(s_h.cuda())
        s_r = self.rel_embedding(s_r.cuda())
        s_t = self.ent_embedding(s_t.cuda())
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_r*(s_h*s_t), dim=1), dim=-1)), dim=-1)
        prior_psl0 = torch.FloatTensor([self._prior_psl]).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros([1]).cuda())).cuda()
        psl_mse = torch.mean(psl_error_each).cuda()
        psl_loss = (psl_mse * self._p_psl).cuda()
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

class U_SLM(nn.Module):
    
    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(U_SLM, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = param.function
        self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim)
        self.mr1 = nn.Embedding(num_embeddings=self.dim, embedding_dim=self.dim)
        self.mr2 = nn.Embedding(num_embeddings=self.dim, embedding_dim=self.dim)

        nn.init.xavier_uniform_(self.ent_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)
        nn.init.xavier_uniform_(self.mr1.weight)
        nn.init.xavier_uniform_(self.mr2.weight)

        self.liner = torch.nn.Linear(1, 1)
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)



    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()

        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        # psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss #+ psl_loss
        return self._A_loss


    def embed(self, h, r, t):
        """Function to get the embedding value.

            Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = self.ent_embedding(h)
        emb_r = self.rel_embedding(r)
        emb_t = self.ent_embedding(t)
        return emb_h, emb_r, emb_t

    def layer(self, h, t):
        """Defines the forward pass layer of the algorithm.

          Args:
              h (Tensor): Head entities ids.
              t (Tensor): Tail entity ids of the triple.
        """
        mr1h = torch.matmul(h, self.mr1.weight)  # h => [m, d], self.mr1 => [d, k]
        mr2t = torch.matmul(t, self.mr2.weight)  # t => [m, d], self.mr2 => [d, k]
        return torch.tanh(mr1h + mr2t)

    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        h_e, r_e, t_e = self.embed(h, r, t)
        n_hn_e, n_rel_hn_e, n_t_e = self.embed(n_hn, n_rel_hn, n_t)
        n_h_e, n_rel_tn_e, n_tn_e = self.embed(n_h, n_rel_tn, n_tn)
        # h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed_complex(h, r, t)
        # n_hn_e_real, n_hn_e_img, n_rel_hn_e_real, n_rel_hn_e_img, n_t_e_real, n_t_e_img = self.embed_complex(n_hn, n_rel_hn, n_t)
        # n_h_e_real, n_h_e_img, n_rel_tn_e_real, n_rel_tn_e_img, n_tn_e_real, n_tn_e_img = self.embed_complex(n_h, n_rel_tn, n_tn)
        ht = self.layer(h_e, t_e)
        htr = torch.unsqueeze(torch.sum(r_e*(ht), dim=1), dim=-1)
        f_prob_h = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze(torch.sum(n_rel_hn_e * self.layer(n_hn_e, n_t_e), dim=2), dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(torch.sum(n_rel_tn_e * self.layer(n_h_e, n_tn_e), dim=2), dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h-w)

        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)

        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)

        this_loss = (torch.sum(((f_score_tn+f_score_hn)/2.0)*self._p_neg+f_score_h)) / self.batch_size
        regularizer = ((torch.sum(torch.square(h_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t_e)) / 2.0) / self.batch_size)

        main_loss = this_loss + self.reg_scale * regularizer

        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.ent_embedding(s_h.cuda())
        s_r = self.rel_embedding(s_r.cuda())
        s_t = self.ent_embedding(s_t.cuda())
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_r*(s_h*s_t), dim=1), dim=-1)), dim=-1)
        prior_psl0 = torch.FloatTensor([self._prior_psl]).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros([1]).cuda())).cuda()
        psl_mse = torch.mean(psl_error_each).cuda()
        psl_loss = (psl_mse * self._p_psl).cuda()
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())



    # def embedding_s(self,h, r, t):
    #     hvec = self.ent_embedding(h.cuda())
    #     rvec = self.rel_embedding(r.cuda())
    #     tvec = self.ent_embedding(t.cuda())
    #     return hvec, rvec, tvec
    # def liner_s(self, h, r, t):
    #     score = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(r*(h*t)), dim=-1)), dim=-1)
    #     return score
    # def liner_sb(self, h, r, t, axis):
    #     score = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(r * (h * t), dim=axis), dim=-1)), dim=-1)
    #     return score

class U_RESCAL(nn.Module):
    
    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(U_RESCAL, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = param.function
        self.gamma = 2
        self.ent_embeddings = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.dim)
        self.rel_matrices = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim * self.dim)

        # self.w = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.liner = torch.nn.Linear(1, 1).cuda()

        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor) -形状为(num_embeddings, embedding_dim)的嵌入中可学习的权值
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_matrices.weight.data)


    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()
        # s_h = torch.tensor(s_h, dtype=torch.int64).cuda()
        # s_r = torch.tensor(s_r, dtype=torch.int64).cuda()
        # s_t = torch.tensor(s_t, dtype=torch.int64).cuda()
        # s_w = torch.tensor(s_w, dtype=torch.float32).cuda()
        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        # psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss #+ psl_loss
        return self._A_loss

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = self.ent_embeddings(h)
        emb_r = self.rel_matrices(r)
        emb_t = self.ent_embeddings(t)
        return emb_h, emb_r, emb_t

    def _calc(self, h, t, r, dim):
        t = t.unsqueeze(-1)
        if dim == 1:
            r = r.view(self.batch_size, self.dim, self.dim)
        elif dim == 2:
            r = r.view(self.batch_size, -1, self.dim, self.dim)
        tr = torch.matmul(r, t)
        tr = tr.squeeze()
        return torch.sum(h * tr, -1)

    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        head, rel, tail = self.embed(h, r, t)
        n_hn, n_rel_hn, n_t = self.embed(n_hn, n_rel_hn, n_t)
        n_h, n_rel_tn, n_tn = self.embed(n_h, n_rel_tn, n_tn)
        score = self._calc(head, tail, rel, 1)
        scorehn = self._calc(n_hn, n_t, n_rel_hn, 2)
        scoretn = self._calc(n_h, n_tn, n_rel_tn, 2)
        htr = torch.unsqueeze(score, dim=-1)
        f_prob_h = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze(scorehn, dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(scoretn, dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h - w)
        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)
        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)
        this_loss = (torch.sum(((f_score_tn + f_score_hn) / 2.0) * self._p_neg + f_score_h)) / self.batch_size
        regularizer = ((torch.sum(torch.square(head)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(tail)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(rel)) / 2.0) / self.batch_size)
        main_loss = this_loss + self.reg_scale * regularizer
        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.sub_embeddings(s_h.cuda())
        s_r = self.rel_embeddings(s_r.cuda())
        s_t = self.obj_embeddings(s_t.cuda())
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_r*(s_h*s_t), dim=1), dim=-1)), dim=-1)
        prior_psl0 = torch.FloatTensor([self._prior_psl]).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros([1]).cuda())).cuda()
        psl_mse = torch.mean(psl_error_each).cuda()
        psl_loss = (psl_mse * self._p_psl).cuda()
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())






class U_CP(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(U_CP, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = param.function
        self.gamma = 2

        self.sub_embeddings = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.dim)
        self.rel_embeddings = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.obj_embeddings = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.dim)
        # self.w = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.liner = torch.nn.Linear(1, 1).cuda()

        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor) -形状为(num_embeddings, embedding_dim)的嵌入中可学习的权值
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)
        nn.init.xavier_uniform_(self.sub_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.obj_embeddings.weight)


    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()
        s_h = torch.tensor(s_h, dtype=torch.int64).cuda()
        s_r = torch.tensor(s_r, dtype=torch.int64).cuda()
        s_t = torch.tensor(s_t, dtype=torch.int64).cuda()
        s_w = torch.tensor(s_w, dtype=torch.float32).cuda()
        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss + psl_loss
        return self._A_loss

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = self.sub_embeddings(h)
        emb_r = self.rel_embeddings(r)
        emb_t = self.obj_embeddings(t)
        return emb_h, emb_r, emb_t

    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        head, rel, tail = self.embed(h, r, t)
        n_hn, n_rel_hn, n_t = self.embed(n_hn, n_rel_hn, n_t)
        n_h, n_rel_tn, n_tn = self.embed(n_h, n_rel_tn, n_tn)


        htr = torch.unsqueeze(torch.sum(rel * (head * tail), dim=1), dim=-1)
        f_prob_h = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze(torch.sum(n_rel_hn * (n_hn * n_t), dim=2), dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(torch.sum(n_rel_tn * (n_h * n_tn), dim=2), dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h - w)
        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)
        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)
        this_loss = (torch.sum(((f_score_tn + f_score_hn) / 2.0) * self._p_neg + f_score_h)) / self.batch_size
        regularizer = ((torch.sum(torch.square(head)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(tail)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(rel)) / 2.0) / self.batch_size)
        main_loss = this_loss + self.reg_scale * regularizer
        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.sub_embeddings(s_h.cuda())
        s_r = self.rel_embeddings(s_r.cuda())
        s_t = self.obj_embeddings(s_t.cuda())
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_r*(s_h*s_t), dim=1), dim=-1)), dim=-1)
        prior_psl0 = torch.FloatTensor([self._prior_psl]).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros([1]).cuda())).cuda()
        psl_mse = torch.mean(psl_error_each).cuda()
        psl_loss = (psl_mse * self._p_psl).cuda()
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

class U_RotatE(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(U_RotatE, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.margin = 2.0
        self.function = param.function
        self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim)
        self.ent_embeddings_imag = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim)
        self.embedding_range = (self.margin + 2.0) / self.dim
        nn.init.xavier_uniform_(self.ent_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_imag.weight)

        self.liner = torch.nn.Linear(1, 1)
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)



    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()
        # s_h = torch.tensor(s_h, dtype=torch.int64).cuda()
        # s_r = torch.tensor(s_r, dtype=torch.int64).cuda()
        # s_t = torch.tensor(s_t, dtype=torch.int64).cuda()
        # s_w = torch.tensor(s_w, dtype=torch.float32).cuda()
        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        # psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss #+ psl_loss
        return self._A_loss

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        pi = 3.14159265358979323846
        h_e_r = self.ent_embedding(h)
        h_e_i = self.ent_embeddings_imag(h)
        r_e_r = self.rel_embedding(r)
        t_e_r = self.ent_embedding(t)
        t_e_i = self.ent_embeddings_imag(t)
        r_e_r = r_e_r / (self.embedding_range / pi)
        r_e_i = torch.sin(r_e_r)
        r_e_r = torch.cos(r_e_r)
        return h_e_r, h_e_i, r_e_r, r_e_i, t_e_r, t_e_i




    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):

        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)
        n_hn_e_real, n_hn_e_img, n_rel_hn_e_real, n_rel_hn_e_img, n_t_e_real, n_t_e_img = self.embed(n_hn, n_rel_hn, n_t)
        n_h_e_real, n_h_e_img, n_rel_tn_e_real, n_rel_tn_e_img, n_tn_e_real, n_tn_e_img = self.embed(n_h, n_rel_tn, n_tn)
        score_r = h_e_real * r_e_real - h_e_img * r_e_img - t_e_real
        score_i = h_e_real * r_e_img + h_e_img * r_e_real - t_e_img
        htr = torch.unsqueeze(self.margin - torch.sum(score_r**2+score_i**2, dim=1), dim=-1)
        f_prob_h = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze(
            self.margin - torch.sum((n_hn_e_real * n_rel_hn_e_real - n_hn_e_img * n_rel_hn_e_img - n_t_e_real) ** 2 +
                                    (n_hn_e_real * n_rel_hn_e_img + n_hn_e_img * n_rel_hn_e_real - n_t_e_img) ** 2,
                                    dim=2), dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(
            self.margin - torch.sum((n_h_e_real * n_rel_tn_e_real - n_h_e_img * n_rel_tn_e_img - n_tn_e_real) ** 2 +
                                    (n_h_e_real * n_rel_tn_e_img + n_h_e_img * n_rel_tn_e_real - n_tn_e_img) ** 2,
                                    dim=2), dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h-w)

        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)

        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)

        this_loss = (torch.sum(((f_score_tn+f_score_hn)/2.0)*self._p_neg+f_score_h)) / self.batch_size
        regularizer = ((torch.sum(torch.square(h_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(h_e_img)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r_e_img)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t_e_real)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t_e_img)) / 2.0) / self.batch_size)
        main_loss = this_loss + self.reg_scale * regularizer

        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.ent_embedding(s_h.cuda())
        s_r = self.rel_embedding(s_r.cuda())
        s_t = self.ent_embedding(s_t.cuda())
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_r*(s_h*s_t), dim=1), dim=-1)), dim=-1)
        prior_psl0 = torch.FloatTensor([self._prior_psl]).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros([1]).cuda())).cuda()
        psl_mse = torch.mean(psl_error_each).cuda()
        psl_loss = (psl_mse * self._p_psl).cuda()
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())



class U_SimplE(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(U_SimplE, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = param.function

        self.ent_head_embeddings = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim)
        self.ent_tail_embeddings = nn.Embedding(num_embeddings=self.num_cons,
                                                embedding_dim=self.dim)
        self.rel_embeddings = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim)
        self.rel_inv_embeddings = nn.Embedding(num_embeddings=self.num_rels,
                                                embedding_dim=self.dim)
        nn.init.xavier_uniform_(self.ent_head_embeddings.weight)
        nn.init.xavier_uniform_(self.ent_tail_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight)
        self.liner = torch.nn.Linear(1, 1).cuda()
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)



    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()
        # s_h = torch.tensor(s_h, dtype=torch.int64).cuda()
        # s_r = torch.tensor(s_r, dtype=torch.int64).cuda()
        # s_t = torch.tensor(s_t, dtype=torch.int64).cuda()
        # s_w = torch.tensor(s_w, dtype=torch.float32).cuda()
        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        # psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss #+ psl_loss
        return self._A_loss

    def embed(self, h, r, t):
        """Function to get the embedding value.

                   Args:
                       h (Tensor): Head entities ids.
                       r (Tensor): Relation ids of the triple.
                       t (Tensor): Tail entity ids of the triple.

                    Returns:
                        Tensors: Returns head, relation and tail embedding Tensors.
                """
        emb_h1 = self.ent_head_embeddings(h)
        emb_h2 = self.ent_head_embeddings(t)
        emb_r1 = self.rel_embeddings(r)
        emb_r2 = self.rel_inv_embeddings(r)
        emb_t1 = self.ent_tail_embeddings(t)
        emb_t2 = self.ent_tail_embeddings(h)
        return emb_h1, emb_h2, emb_r1, emb_r2, emb_t1, emb_t2

    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        h1_e, h2_e, r1_e, r2_e, t1_e, t2_e = self.embed(h, r, t)
        hn_h1_e, hn_h2_e, hn_r1_e, hn_r2_e, hn_t1_e, hn_t2_e = self.embed(n_hn, n_rel_hn, n_t)
        tn_h1_e, tn_h2_e, tn_r1_e, tn_r2_e, tn_t1_e, tn_t2_e = self.embed(n_h, n_rel_tn, n_tn)

        init = (torch.sum((h1_e * t1_e) * r1_e, 1) + torch.sum((h2_e * t2_e) * r2_e, 1)) / 2.0
        htr = torch.unsqueeze(init, dim=-1)
        f_prob_h = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze((torch.sum(hn_h1_e * hn_r1_e * hn_t1_e, 2) + torch.sum(hn_h2_e * hn_r2_e * hn_t2_e, 2)) / 2.0,
                                                             dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze((torch.sum(tn_h1_e * tn_r1_e * tn_t1_e, 2) + torch.sum(tn_h2_e * tn_r2_e * tn_t2_e, 2)) / 2.0,
                                                             dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h-w)

        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)

        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)

        this_loss = (torch.sum(((f_score_tn+f_score_hn)/2.0)*self._p_neg+f_score_h)) / self.batch_size
        regularizer = ((torch.sum(torch.square(h1_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(h2_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r1_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(r2_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t1_e)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(t2_e)) / 2.0) / self.batch_size)
        main_loss = this_loss + self.reg_scale * regularizer

        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.ent_embedding(s_h.cuda())
        s_r = self.rel_embedding(s_r.cuda())
        s_t = self.ent_embedding(s_t.cuda())
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_r*(s_h*s_t), dim=1), dim=-1)), dim=-1)
        prior_psl0 = torch.FloatTensor([self._prior_psl]).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros([1]).cuda())).cuda()
        psl_mse = torch.mean(psl_error_each).cuda()
        psl_loss = (psl_mse * self._p_psl).cuda()
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

class UH(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(UH, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = param.function
        self.gamma = 2
        # self.margin = nn.Parameter(torch.Tensor([self.gamma]))
        # self.margin.requires_grad = False
        self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.w = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.liner = torch.nn.Linear(1, 1).cuda()
        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor) -形状为(num_embeddings, embedding_dim)的嵌入中可学习的权值
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)
        nn.init.xavier_uniform_(self.ent_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)
        nn.init.xavier_uniform_(self.w.weight)



    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()
        # s_h = torch.tensor(s_h, dtype=torch.int64).cuda()
        # s_r = torch.tensor(s_r, dtype=torch.int64).cuda()
        # s_t = torch.tensor(s_t, dtype=torch.int64).cuda()
        # s_w = torch.tensor(s_w, dtype=torch.float32).cuda()
        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        # psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss #+ psl_loss
        return self._A_loss

    def embed(self, h, r, t):

        emb_h = self.ent_embedding(h)
        emb_r = self.rel_embedding(r)
        emb_t = self.ent_embedding(t)
        proj_vec = self.w(r)

        emb_h = self._projection(emb_h, proj_vec)
        emb_t = self._projection(emb_t, proj_vec)

        return emb_h, emb_r, emb_t

    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        head, rel, tail = self.embed(h, r, t)
        n_hn, n_rel_hn, n_t = self.embed(n_hn, n_rel_hn, n_t)
        n_h, n_rel_tn, n_tn =self.embed(n_h, n_rel_tn, n_tn)

        htr = torch.unsqueeze(torch.sum(rel*(head * tail), dim=1), dim=-1)
        f_prob_h = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze(torch.sum(n_rel_hn * (n_hn * n_t), dim=2), dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(torch.sum(n_rel_tn * (n_h * n_tn), dim=2), dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h-w)
        # s = torch.norm((n_rel_hn + n_hn) - n_t, p=1, dim=-1)

        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)

        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)
        regularizer = ((torch.sum(torch.square(head)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(rel)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(tail)) / 2.0) / self.batch_size)
        # this_loss = (f_score_h - (f_score_hn+f_score_tn)/2.0).mean()
        this_loss = (torch.sum(((f_score_tn+f_score_hn)/2.0)*self._p_neg+f_score_h)) / self.batch_size
        main_loss = this_loss + self.reg_scale * regularizer
        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.ent_embedding(s_h.cuda())
        s_r = self.rel_embedding(s_r.cuda())
        s_t = self.ent_embedding(s_t.cuda())
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_r*(s_h*s_t), dim=1), dim=-1)), dim=-1)
        prior_psl0 = torch.FloatTensor([self._prior_psl]).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros([1]).cuda())).cuda()
        psl_mse = torch.mean(psl_error_each).cuda()
        psl_loss = (psl_mse * self._p_psl).cuda()
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    @staticmethod
    def _projection(emb_e, proj_vec):
        """Calculates the projection of entities"""

        return emb_e - torch.sum(emb_e * proj_vec, dim=-1, keepdims=True) * proj_vec


class UD(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(UD, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = param.function
        self.gamma = 2
        # self.margin = nn.Parameter(torch.Tensor([self.gamma]))
        # self.margin.requires_grad = False
        self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.ent_mappings = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.dim)
        self.rel_mappings = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.liner = torch.nn.Linear(1, 1)

        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor) -形状为(num_embeddings, embedding_dim)的嵌入中可学习的权值
        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)
        nn.init.xavier_uniform_(self.ent_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)
        nn.init.xavier_uniform_(self.ent_mappings.weight)
        nn.init.xavier_uniform_(self.rel_mappings.weight)


    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()

        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        # psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss #+ psl_loss
        return self._A_loss

    def embed(self, h, r, t):
        emb_h = self.ent_embedding(h)
        emb_r = self.rel_embedding(r)
        emb_t = self.ent_embedding(t)

        h_m = self.ent_mappings(h)
        r_m = self.rel_mappings(r)
        t_m = self.ent_mappings(t)

        emb_h = self._projection(emb_h, h_m, r_m)
        emb_t = self._projection(emb_t, t_m, r_m)
        return emb_h, emb_r, emb_t

    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        head, rel, tail = self.embed(h, r, t)
        n_hn, n_rel_hn, n_t = self.embed(n_hn, n_rel_hn, n_t)
        n_h, n_rel_tn, n_tn =self.embed(n_h, n_rel_tn, n_tn)
        htr = torch.unsqueeze(torch.sum((tail*head) * rel, dim=1), dim=-1)
        f_prob_h = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze(torch.sum((n_t * n_hn) * n_rel_hn, dim=2), dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(torch.sum((n_tn * n_h) * n_rel_tn, dim=2), dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h-w)
        # s = torch.norm((n_rel_hn + n_hn) - n_t, p=1, dim=-1)

        f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)

        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)
        regularizer = ((torch.sum(torch.square(head)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(rel)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(tail)) / 2.0) / self.batch_size)
        # this_loss = (f_score_h - (f_score_hn+f_score_tn)/2.0).mean()
        this_loss = (torch.sum(((f_score_tn+f_score_hn)/2.0)*self._p_neg+f_score_h)) / self.batch_size
        main_loss = this_loss + self.reg_scale * regularizer
        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.ent_embedding(s_h.cuda())
        s_r = self.rel_embedding(s_r.cuda())
        s_t = self.ent_embedding(s_t.cuda())
        psl_prob = torch.squeeze(self.liner(torch.unsqueeze(torch.sum(s_r*(s_h*s_t), dim=1), dim=-1)), dim=-1)
        prior_psl0 = torch.FloatTensor([self._prior_psl]).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros([1]).cuda())).cuda()
        psl_mse = torch.mean(psl_error_each).cuda()
        psl_loss = (psl_mse * self._p_psl).cuda()
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    @staticmethod
    def _projection(emb_e, emb_m, proj_vec):

        return emb_e + torch.sum(emb_e * emb_m, dim=-1, keepdims=True) * proj_vec


class UKGE(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        super(UKGE, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = param.function
        self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.liner = torch.nn.Linear(1, 1).cuda()
        self.__data_init()

    def __data_init(self):

        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)
        nn.init.xavier_uniform_(self.ent_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)



    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()
        s_h = torch.tensor(s_h, dtype=torch.int64).cuda()
        s_r = torch.tensor(s_r, dtype=torch.int64).cuda()
        s_t = torch.tensor(s_t, dtype=torch.int64).cuda()
        s_w = torch.tensor(s_w, dtype=torch.float32).cuda()
        main_loss = self.main_loss(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn)
        psl_loss = self.define_psl_loss(s_h, s_r, s_t, s_w)
        self._A_loss = main_loss + psl_loss
        return self._A_loss



    def main_loss(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):
        head = self.ent_embedding(h)
        rel = self.rel_embedding(r)
        tail = self.ent_embedding(t)
        n_hn = self.ent_embedding(n_hn)
        n_rel_hn = self.rel_embedding(n_rel_hn)
        n_t = self.ent_embedding(n_t)
        n_h = self.ent_embedding(n_h)
        n_rel_tn = self.rel_embedding(n_rel_tn)
        n_tn = self.ent_embedding(n_tn)

        htr = torch.unsqueeze(torch.sum(rel * (head * tail), dim=1), dim=-1)
        f_prob_h = self.liner(htr)
        f_prob_hn = self.liner(torch.unsqueeze(torch.sum(n_rel_hn * (n_hn * n_t), dim=2), dim=-1))
        f_prob_tn = self.liner(torch.unsqueeze(torch.sum(n_rel_tn * (n_h * n_tn), dim=2), dim=-1))
        if self.function == 'logi':
            f_prob_h = torch.sigmoid(f_prob_h)
            f_prob_hn = torch.sigmoid(f_prob_hn)
            f_prob_tn = torch.sigmoid(f_prob_tn)
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.square(f_prob_h - w)
        f_prob_hn = torch.squeeze(f_prob_hn,dim=-1)
        f_score_hn = torch.mean(torch.square(f_prob_hn), dim=1)
        f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
        f_score_tn = torch.mean(torch.square(f_prob_tn), dim=1)
        this_loss = (torch.sum(((f_score_tn + f_score_hn) / 2.0) * self._p_neg + f_score_h)) / self.batch_size
        regularizer = ((torch.sum(torch.square(head)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(tail)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(rel)) / 2.0) / self.batch_size)
        main_loss = this_loss + self.reg_scale * regularizer
        return main_loss
    def define_psl_loss(self, s_h, s_r, s_t, s_w):
        s_h = self.ent_embedding(s_h.cuda())
        s_r = self.rel_embedding(s_r.cuda())
        s_t = self.ent_embedding(s_t.cuda())
        psl_prob = self.liner(torch.unsqueeze(torch.sum(s_r * (s_h * s_t), dim=1), dim=-1))
        if self.function == 'logi':
            psl_prob = torch.sigmoid(psl_prob)
        psl_prob = torch.squeeze(psl_prob, dim=-1)

        prior_psl0 = torch.tensor(self._prior_psl).cuda()
        psl_error_each = torch.square(torch.maximum(s_w + prior_psl0 - psl_prob, torch.zeros(1).cuda()))
        psl_mse = torch.mean(psl_error_each)
        psl_loss = psl_mse * self._p_psl
        return psl_loss
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())









class U_ANALOGY_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = U_ANALOGY(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class U_ComplEx_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = U_ComplEx(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class U_SLM_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = U_SLM(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class U_RESCAL_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = U_RESCAL(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class U_CP_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = U_CP(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class U_RotatE_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = U_RotatE(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class U_SimplE_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = U_SimplE(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class UH_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = UH(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class UD_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):

        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = UD(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim

class UKGE_PT:
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, lr):
        # PTParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, reg_scale)
        # self.build()
        self.nr = num_rels
        self.nc = num_cons
        self.dim = dim
        self.bs = batch_size
        self.n_per_p = neg_per_positive
        self.rs = reg_scale
        self.p_n = p_neg
        self.lr = lr
    def build(self):
        self.model = UKGE(self.nr, self.nc, self.dim, self.bs, self.n_per_p, self.rs, self.p_n).cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optim







