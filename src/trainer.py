''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from os.path import join

from src import param

import sys

if '../src' not in sys.path:
    sys.path.append('../src')

import numpy as np
# import tensorflow as tf
import time
from src.data import BatchLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import operator
from src.list import ModelList
from src.models import U_ANALOGY_PT, U_ComplEx_PT, U_SLM_PT, U_RESCAL_PT, \
    U_CP_PT,U_RotatE_PT,U_SimplE_PT,UH_PT,UD_PT,UKGE_PT
from src.testers import U_ANALOGY_Tester, U_ComplEx_Tester, U_SLM_Tester, U_RESCAL_Tester,U_CP_Tester,\
    U_RotatE_Tester,U_SimplE_Tester,UH_Tester,UD_Tester,UKGE_Tester


class Trainer(object):
    def __init__(self):
        self.batch_size = 128
        self.dim = 128
        self.this_data = None
        self.tf_parts = None
        self.file_val = ""
        self.L1 = False


    def build(self, data_obj, save_dir, lr, modelname):
        """
        All files are stored in save_dir.
        output files:
        1. tf model
        2. this_data (Data())
        3. training_loss.csv, val_loss.csv
        :param model_save: filename for model
        :param data_save: filename for self.this_data
        :param knn_neg: use kNN negative sampling
        :return:
        """
        self.verbose = param.verbose  # print extra information
        self.this_data = data_obj
        self.dim = self.this_data.dim = param.dim
        self.batch_size = self.this_data.batch_size = param.batch_size
        self.neg_per_positive = param.neg_per_pos
        self.reg_scale = param.reg_scale
        self.modelname = modelname
        self.batchloader = BatchLoader(self.this_data, self.batch_size, self.neg_per_positive)

        self.p_neg = param.p_neg
        self.p_psl = param.p_psl

        # paths for saving
        self.save_dir = save_dir
        self.train_loss_path = join(save_dir, 'trainig_loss.csv')
        self.val_loss_path = join(save_dir, 'val_loss.csv')

        print('Now using model: ', param.whichmodel)

        self.whichmodel = param.whichmodel
        self.lr = lr
        self.build_pt_parts()  # could be overrided


    def build_pt_parts(self,):

        if self.whichmodel == ModelList.U_ComplEx:
            self.pt_parts = U_ComplEx_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg,
                                         reg_scale=self.reg_scale, lr=self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = U_ComplEx_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())

        elif self.whichmodel == ModelList.U_ANALOGY:
            self.pt_parts = U_ANALOGY_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = U_ANALOGY_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())
        elif self.whichmodel == ModelList.U_SLM:
            self.pt_parts = U_SLM_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = U_SLM_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())
        elif self.whichmodel == ModelList.U_RESCAL:
            self.pt_parts = U_RESCAL_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = U_RESCAL_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())
        elif self.whichmodel == ModelList.U_CP:
            self.pt_parts = U_CP_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = U_CP_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())
        elif self.whichmodel == ModelList.U_CP:
            self.pt_parts = U_CP_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = U_CP_Tester(self.modelname,
                                         num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons())
        elif self.whichmodel == ModelList.U_RotatE:
            self.pt_parts = U_RotatE_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = U_RotatE_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())

        elif self.whichmodel == ModelList.U_SimplE:
            self.pt_parts = U_SimplE_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = U_SimplE_Tester(self.modelname,
                                             num_rels=self.this_data.num_rels(),
                                             num_cons=self.this_data.num_cons())
        elif self.whichmodel == ModelList.UH:
            self.pt_parts = UH_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = UH_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())

        elif self.whichmodel == ModelList.UD:
            self.pt_parts = UD_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = UD_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())
        elif self.whichmodel == ModelList.UKGE:
            self.pt_parts = UKGE_PT(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale,lr = self.lr)
            self.model, self.optimizer = self.pt_parts.build()
            self.validator = UKGE_Tester(self.modelname, 
                                              num_rels=self.this_data.num_rels(),
                                              num_cons=self.this_data.num_cons())

    def train(self, epochs=20, save_every_epoch=10, lr=0.001, data_dir=""):

        num_batch = self.this_data.triples.shape[0] // self.batch_size
        print('Number of batches per epoch: %d' % num_batch)
        checkpoint = self.save_dir + '/checkpoint/'
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        train_losses = []  # [[every epoch, loss]]
        val_losses = []  # [[saver epoch, loss]]

        for epoch in range(1, epochs + 1):
            self.loss = 0.0



            epoch_batches = self.batchloader.gen_batch(forever=True)
            epoch_loss = []

            for batch_id in range(num_batch):
                self.model.train()
                batch = next(epoch_batches)
                A_h_index, A_r_index, A_t_index, A_w, \
                A_neg_hn_index, A_neg_rel_hn_index, \
                A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch

                # time00 = time.time()
                soft_h_index, soft_r_index, soft_t_index, soft_w_index = self.batchloader.gen_psl_samples()
                self.update_triple_embedding(A_h_index, A_r_index, A_t_index, A_w,
                                             A_neg_hn_index, A_neg_rel_hn_index,
                                             A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index,
                                             soft_h_index, soft_r_index, soft_t_index, soft_w_index)
            mean_loss = self.loss / num_batch
            train_losses.append([epoch, mean_loss])
            print("Loss of epoch %d = %s" % (epoch, mean_loss))
            if epoch % save_every_epoch == 0:

                #save model

                self.model.save_checkpoint(checkpoint+str(epoch)+'KE.ckpt')
                mse, mse_pos, mse_neg, mae, mae_pos, mae_neg, mean_ndcg, mean_exp_ndcg = self.get_val_loss(epoch, dir=checkpoint)
                val_losses.append([epoch, mse, mae, mse_pos, mse_neg, mae_pos, mae_neg, mean_ndcg, mean_exp_ndcg])

                # save and print metrics
                self.save_loss(train_losses, self.train_loss_path, columns=['epoch', 'training_loss'])
                self.save_loss(val_losses, self.val_loss_path,
                               columns=['val_epoch', 'mse(10^-2)', 'mae(10^-2)', 'mse_pos', 'mse_neg', 'mae_pos', 'mae_neg', 'ndcg(linear)', 'ndcg(exp)'])
                # pred_thres = np.arange(0, 1, 0.05)
                # scores, P, R, f1, Acc = self.validator.classify_triples(0.85, pred_thres)
                # print('\n', np.max(f1), '\n', np.max(Acc), '\n')
                # train_data = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', header=None,
                #                          names=['v1', 'relation', 'v2', 'w'])
                # test_X, precision, recall, F1, accu, P, R = self.validator.decision_tree_classify(0.85, train_data)
                # print('\n')
                # print(F1)
                # print('\n')
                # print(accu)
                # print('\n')






    def update_triple_embedding(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w):
        self.optimizer.zero_grad()
        loss = self.model(h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn, s_h, s_r, s_t, s_w)
        self.loss += loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss



    def get_val_loss(self, epoch,dir):
        # validation error

        self.validator.build_by_var(self.this_data.val_triples, epoch, self.this_data, dir)

        if not hasattr(self.validator, 'hr_map'):
            self.validator.load_hr_map(param.data_dir(), 'test.tsv', ['train.tsv', 'val.tsv',
                                                                          'test.tsv'])
        if not hasattr(self.validator, 'hr_map_sub'):
            hr_map200 = self.validator.get_fixed_hr(n=200)  # use smaller size for faster validation
        else:
            hr_map200 = self.validator.hr_map_sub

        mean_ndcg, mean_exp_ndcg = self.validator.mean_ndcg(hr_map200)

        # metrics: mse
        mse_pos = self.validator.get_mse(save_dir=self.save_dir, epoch=epoch, toprint=self.verbose)
        mse_neg = self.validator.get_mse_neg(self.neg_per_positive)
        mae_pos = self.validator.get_mae(save_dir=self.save_dir, epoch=epoch)
        mae_neg = self.validator.get_mae_neg(self.neg_per_positive)
        mse = ((mse_pos+mse_neg)/2)*100
        mae = ((mae_pos+mae_neg)/2)*100
        return mse, mse_pos, mse_neg, mae, mae_pos, mae_neg, mean_ndcg, mean_exp_ndcg

    def save_loss(self, losses, filename, columns):
        df = pd.DataFrame(losses, columns=columns)
        print(df.tail(5))
        df.to_csv(filename, index=False)