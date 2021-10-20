''' Module for held-out test.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import tensorflow as tf
from numpy import linalg as LA
import heapq as HP
import pandas as pd
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from scipy.special import expit as sigmoid

import sys

if '../' not in sys.path:  # src folder
    sys.path.append('../')

from os.path import join
import data
import time
import pickle
import random
import torch
import sklearn
from sklearn import tree
from src import param


# This class is used to load and combine a TF_Parts and a Data object, and provides some useful methods for training
class Tester(object):
    class IndexScore:
        """
        The score of a tail when h and r is given.
        It's used in the ranking task to facilitate comparison and sorting.
        Print w as 3 digit precision float.
        """

        def __init__(self, index, score):
            self.index = index
            self.score = score

        def __lt__(self, other):
            return self.score < other.score

        def __repr__(self):
            # return "(index: %d, w:%.3f)" % (self.index, self.score)
            return "(%d, %.3f)" % (self.index, self.score)

        def __str__(self):
            return "(index: %d, w:%.3f)" % (self.index, self.score)

    def __init__(self, modelname, num_rels, num_cons):
        self.tf_parts = None
        self.this_data = None
        if modelname == "uanalogy":
            self.vec_c = np.array([0])
            self.vec_r = np.array([0])
            self.vec_c_real = np.array([0])
            self.vec_r_real = np.array([0])
            self.vec_c_img = np.array([0])
            self.vec_r_img = np.array([0])
        elif modelname == "ucomplex":
            self.vec_c_real = np.array([0])
            self.vec_r_real = np.array([0])
            self.vec_c_img = np.array([0])
            self.vec_r_img = np.array([0])
        elif modelname == "uslm":
            self.vec_c = np.array([0])
            self.vec_r = np.array([0])
            self.vec_mr1 = np.array([0])
            self.vec_mr2 = np.array([0])
        elif modelname == "urescal":
            self.vec_c = np.array([0])
            self.vec_r = np.array([0])
        elif modelname =="ucp":
            self.vec_h = np.array([0])
            self.vec_r = np.array([0])
            self.vec_t = np.array([0])
        elif modelname == "urotate":
            self.vec_c = np.array([0])
            self.vec_r = np.array([0])
            self.vec_c_img = np.array([0])
        elif modelname == "usimple":
            self.vec_c_h = np.array([0])
            self.vec_r = np.array([0])
            self.vec_c_t = np.array([0])
            self.vec_r_inv = np.array([0])
        elif modelname == "uh":
            self.vec_c = np.array([0])
            self.vec_r = np.array([0])
            self.vec_w = np.array([0])
        elif modelname == "ukge":
            self.vec_c = np.array([0])
            self.vec_r = np.array([0])
            self.vec_w = np.array([0])
        elif modelname == "ud" :
            self.vec_c = np.array([0])
            self.vec_r = np.array([0])
            self.vec_cm = np.array([0])
            self.vec_rm = np.array([0])

        # below for test data
        self.cons = num_cons
        self.rels = num_rels
        self.test_triples = np.array([0])
        self.test_triples_group = {}

    # completed by child class
    def build_by_file(self, test_data_file, model_dir, model_filename='xcn-distmult.ckpt', data_filename='xc-data.bin'):
        # load the saved Data()
        self.this_data = data.Data()
        data_save_path = join(model_dir, data_filename)
        self.this_data.load(data_save_path)

        # load testing data
        self.load_test_data(test_data_file)

        self.model_dir = model_dir  # used for saving

    # abstract method
    # def build_by_var(self, test_data, tf_model, this_data, sess):
    #     raise NotImplementedError("Fatal Error: This model' tester didn't implement its build_by_var() function!")


    def load_hr_map(self, data_dir, hr_base_file, supplement_t_files, splitter='\t', line_end='\n'):
        """
        Initialize self.hr_map.
        Load self.hr_map={h:{r:t:w}}}, not restricted to test data
        :param hr_base_file: Get self.hr_map={h:r:{t:w}}} from the file.
        :param supplement_t_files: Add t(only t) to self.hr_map. Don't add h or r.
        :return:
        """
        self.hr_map = {}
        with open(join(data_dir, hr_base_file)) as f:
            for line in f:
                line = line.rstrip(line_end).split(splitter)
                h = self.this_data.con_str2index(line[0])
                r = self.this_data.rel_str2index(line[1])
                t = self.this_data.con_str2index(line[2])
                w = float(line[3])
                # construct hr_map
                if self.hr_map.get(h) == None:
                    self.hr_map[h] = {}
                if self.hr_map[h].get(r) == None:
                    self.hr_map[h][r] = {t: w}
                else:
                    self.hr_map[h][r][t] = w

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

        for file in supplement_t_files:
            with open(join(data_dir, file)) as f:
                for line in f:
                    line = line.rstrip(line_end).split(splitter)
                    h = self.this_data.con_str2index(line[0])
                    r = self.this_data.rel_str2index(line[1])
                    t = self.this_data.con_str2index(line[2])
                    w = float(line[3])

                    # update hr_map
                    if h in self.hr_map and r in self.hr_map[h]:
                        self.hr_map[h][r][t] = w

    def save_hr_map(self, outputfile):
        """
        Print to file for debugging. (not applicable for reloading)
        Prerequisite: self.hr_map has been loaded.
        :param outputfile:
        :return:
        """
        if self.hr_map is None:
            raise ValueError("Tester.hr_map hasn't been loaded! Use Tester.load_hr_map() to load it.")

        with open(outputfile, 'w') as f:
            for h in self.hr_map:
                for r in self.hr_map[h]:
                    tw_truth = self.hr_map[h][r]  # {t:w}
                    tw_list = [self.IndexScore(t, w) for t, w in tw_truth.items()]
                    tw_list.sort(reverse=True)  # descending on w
                    f.write('h: %d, r: %d\n' % (h, r))
                    f.write(str(tw_list) + '\n')

    def load_test_data(self, filename, splitter='\t', line_end='\n'):
        num_lines = 0
        triples = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 4:
                continue
            num_lines += 1
            h = self.this_data.con_str2index(line[0])
            r = self.this_data.rel_str2index(line[1])
            t = self.this_data.con_str2index(line[2])
            w = float(line[3])
            if h is None or r is None or t is None or w is None:
                continue
            triples.append([h, r, t, w])

            # add to group
            if self.test_triples_group.get(r) == None:
                self.test_triples_group[r] = [(h, r, t, w)]
            else:
                self.test_triples_group[r].append((h, r, t, w))

        # Note: test_triples will be a np.float64 array! (because of the type of w)
        # Take care of the type of hrt when unpacking.
        self.test_triples = np.array(triples)

        print("Loaded test data from %s, %d out of %d." % (filename, len(triples), num_lines))
        # print("Rel each cat:", self.rel_num_cases)





    def con_str2vec(self, str):
        this_index = self.this_data.con_str2index(str)
        if this_index == None:
            return None
        return self.vec_c[this_index]

    def rel_str2vec(self, str):
        this_index = self.this_data.rel_str2index(str)
        if this_index == None:
            return None
        return self.vec_r[this_index]

    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return

        def __lt__(self, other):
            return self.dist > other.dist

    def con_index2str(self, str):
        return self.this_data.con_index2str(str)

    def rel_index2str(self, str):
        return self.this_data.rel_index2str(str)

    # input must contain a pool of pre- or post-projected vecs. return a list of indices and dist
    def kNN(self, vec, vec_pool, topk=10, self_id=None):
        q = []
        for i in range(len(vec_pool)):
            # skip self
            if i == self_id:
                continue
            dist = np.dot(vec, vec_pool[i])
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                # indeed it fetches the biggest
                # as the index_dist "lt" is defined as larger dist
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist < dist:
                    HP.heapreplace(q, self.index_dist(i, dist))
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst

    # input must contain a pool of pre- or post-projected vecs. return a list of indices and dist. rank an index in a vec_pool from
    def rank_index_from(self, vec, vec_pool, index, self_id=None):
        dist = np.dot(vec, vec_pool[index])
        rank = 1
        for i in range(len(vec_pool)):
            if i == index or i == self_id:
                continue
            if dist < np.dot(vec, vec_pool[i]):
                rank += 1
        return rank

    def rel_cat_id(self, r):
        if r in self.relnn:
            return 3
        elif r in self.rel1n:
            return 1
        elif r in self.reln1:
            return 2
        else:
            return 0

    def dissimilarity(self, h, r, t):
        h_vec = self.vec_c[h]
        t_vec = self.vec_c[t]
        r_vec = self.vec_r[r]
        return np.dot(r_vec, np.multiply(h_vec, t_vec))

    def dissimilarity2(self, h, r, t):
        # h_vec = self.vec_c[h]
        # t_vec = self.vec_c[t]
        r_vec = self.vec_r[r]
        return np.dot(r_vec, np.multiply(h, t))

    def get_info(self, triple):
        """
        convert the float h, r, t to int, and return
        :param triple: triple: np.array[4], dtype=np.float64: h,r,t,w
        :return: h, r, t(index), w(float)
        """
        h_, r_, t_, w = triple  # h_, r_, t_: float64
        return int(h_), int(r_), int(t_), w


    # Abstract method. Different scoring function for different models.
    def get_score(self, h, r, t):
        raise NotImplementedError("get_score() is not defined in this model's tester")

    # Abstract method. Different scoring function for different models.
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        raise NotImplementedError("get_score_batch() is not defined in this model's tester")

    def get_bound_score(self, h, r, t):
        # for most models, just return the original score
        # may be overwritten by child class
        return self.get_score(h, r, t)

    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # for non-rect models, just return the original score
        # rect models will override it
        return self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)

    def get_mse(self, toprint=False, save_dir='', epoch=0):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        if param.function == 'rect':
            scores = self.bound_score(scores)
        mse = np.sum(np.square(scores - w_batch))

        mse = mse / N

        return mse

    def get_mae(self, verbose=False, save_dir='', epoch=0):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        if param.function == 'rect':
            scores = self.bound_score(scores)
        mae = np.sum(np.absolute(scores - w_batch))

        mae = mae / N
        return mae

    def get_mse_neg(self, neg_per_positive):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # negative samples
        # (score - 0)^2
        all_neg_hn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "h")
        all_neg_tn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "t")
        neg_hn_batch, neg_rel_hn_batch, \
        neg_t_batch, neg_h_batch, \
        neg_rel_tn_batch, neg_tn_batch \
            = all_neg_hn_batch[:, :, 0].astype(int), \
              all_neg_hn_batch[:, :, 1].astype(int), \
              all_neg_hn_batch[:, :, 2].astype(int), \
              all_neg_tn_batch[:, :, 0].astype(int), \
              all_neg_tn_batch[:, :, 1].astype(int), \
              all_neg_tn_batch[:, :, 2].astype(int)
        scores_hn = self.get_score_batch(neg_hn_batch, neg_rel_hn_batch, neg_t_batch, isneg2Dbatch=True)
        scores_tn = self.get_score_batch(neg_h_batch, neg_rel_tn_batch, neg_tn_batch, isneg2Dbatch=True)
        if param.function == 'rect':
            scores_hn = self.bound_score(scores_hn)
            scores_tn = self.bound_score(scores_tn)

        mse_hn = np.sum(np.mean(np.square(scores_hn - 0), axis=1)) / N
        mse_tn = np.sum(np.mean(np.square(scores_tn - 0), axis=1)) / N

        mse = (mse_hn + mse_tn) / 2
        return mse

    def get_t_ranks(self, h, r, ts):
        """
        Given some t index, return the ranks for each t
        :return:
        """
        # prediction
        scores = np.array([self.get_score(h, r, t) for t in ts])  # predict scores for t from ground truth

        ranks = np.ones(len(ts), dtype=int)  # initialize rank as all 1

        N = self.cons  # pool of t: all concept vectors
        for i in range(N):  # compute scores for all concept vectors as t
            score_i = self.get_score(h, r, i)
            rankplus = (scores < score_i).astype(int)  # rank+1 if score<score_i
            ranks += rankplus

        return ranks





    def get_mae_neg(self, neg_per_positive):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # negative samples
        # (score - 0)^2
        all_neg_hn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "h")
        all_neg_tn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "t")
        neg_hn_batch, neg_rel_hn_batch, \
        neg_t_batch, neg_h_batch, \
        neg_rel_tn_batch, neg_tn_batch \
            = all_neg_hn_batch[:, :, 0].astype(int), \
              all_neg_hn_batch[:, :, 1].astype(int), \
              all_neg_hn_batch[:, :, 2].astype(int), \
              all_neg_tn_batch[:, :, 0].astype(int), \
              all_neg_tn_batch[:, :, 1].astype(int), \
              all_neg_tn_batch[:, :, 2].astype(int)
        scores_hn = self.get_score_batch(neg_hn_batch, neg_rel_hn_batch, neg_t_batch, isneg2Dbatch=True)
        scores_tn = self.get_score_batch(neg_h_batch, neg_rel_tn_batch, neg_tn_batch, isneg2Dbatch=True)
        if param.function == 'rect':
            scores_hn = self.bound_score(scores_hn)
            scores_tn = self.bound_score(scores_tn)
        mae_hn = np.sum(np.mean(np.absolute(scores_hn - 0), axis=1)) / N
        mae_tn = np.sum(np.mean(np.absolute(scores_tn - 0), axis=1)) / N

        mae_neg = (mae_hn + mae_tn) / 2
        return mae_neg



    def pred_top_k_tail(self, k, h, r):
        """
        Predict top k tail.
        The #returned items <= k.
        Consider add tail_pool to limit the range of searching.
        :param k: how many results to return
        :param h: index of head
        :param r: index of relation
        :return:
        """
        q = []  # min heap
        N = self.vec_c.shape[0]  # the total number of concepts
        for t_idx in range(N):
            score = self.get_score(h, r, t_idx)
            if len(q) < k:
                HP.heappush(q, self.IndexScore(t_idx, score))
            else:
                tmp = q[0]  # smallest score
                if tmp.score < score:
                    HP.heapreplace(q, self.IndexScore(t_idx, score))

        indices = np.zeros(len(q), dtype=int)
        scores = np.ones(len(q), dtype=float)
        i = len(q) - 1  # largest score first
        while len(q) > 0:
            item = HP.heappop(q)
            indices[i] = item.index
            scores[i] = item.score
            i -= 1

        return indices, scores



    def ndcg(self, h, r, tw_truth):
        """
        Compute nDCG(normalized discounted cummulative gain)
        sum(score_ground_truth / log2(rank+1)) / max_possible_dcg
        :param tw_truth: [IndexScore1, IndexScore2, ...], soreted by IndexScore.score descending
        :return:
        """
        # prediction
        ts = [tw.index for tw in tw_truth]
        ranks = self.get_t_ranks(h, r, ts)

        # linear gain
        gains = np.array([tw.score for tw in tw_truth])
        discounts = np.log2(ranks + 1)
        discounted_gains = gains / discounts
        dcg = np.sum(discounted_gains)  # discounted cumulative gain
        # normalize
        max_possible_dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain

        # exponential gain
        exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
        exp_discounted_gains = exp_gains / discounts
        exp_dcg = np.sum(exp_discounted_gains)
        # normalize
        exp_max_possible_dcg = np.sum(
            exp_gains / np.log2(np.arange(len(exp_gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

        return ndcg, exp_ndcg

    def mean_ndcg(self, hr_map):
        """
        :param hr_map: {h:{r:{t:w}}}
        :return:
        """
        ndcg_sum = 0  # nDCG with linear gain
        exp_ndcg_sum = 0  # nDCG with exponential gain
        count = 0

        t0 = time.time()

        # debug ndcg
        res = []  # [(h,r,tw_truth, ndcg)]

        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                tw_truth = [self.IndexScore(t, w) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ndcg, exp_ndcg = self.ndcg(h, r, tw_truth)  # nDCG with linear gain and exponential gain
                ndcg_sum += ndcg
                exp_ndcg_sum += exp_ndcg
                count += 1
                # if count % 100 == 0:
                #     print('Processed %d, time %s' % (count, (time.time() - t0)))
                #     print('mean ndcg (linear gain) now: %f' % (ndcg_sum / count))
                #     print('mean ndcg (exponential gain) now: %f' % (exp_ndcg_sum / count))

                # # debug
                # ranks = self.get_t_ranks(h, r, [tw.index for tw in tw_truth])
                # res.append((h,r,tw_truth, ndcg, ranks))

        return ndcg_sum / count, exp_ndcg_sum / count


    def classify_triples(self, confT, plausTs):
        """
        Classify high-confidence relation facts
        :param confT: the threshold of ground truth confidence score
        :param plausTs: the list of proposed thresholds of computed plausibility score
        :return:
        """
        test_triples = self.test_triples

        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]

        # ground truth
        high_gt = set(np.squeeze(np.argwhere(w_batch > confT)))  # positive
        low_gt = set(np.squeeze(np.argwhere(w_batch <= confT)))  # negative

        P = []
        R = []
        Acc = []

        # prediction
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        print('The mean of prediced scores: %f' % np.mean(scores))
        # pred_thres = np.arange(0, 1, 0.05)
        for pthres in plausTs:

            high_pred = set(np.squeeze(np.argwhere(scores > pthres)))
            low_pred = set(np.squeeze(np.argwhere(scores <= pthres)))

            # precision-recall
            TP = high_gt & high_pred  # union intersection
            if len(high_pred) == 0:
                precision = 1
            else:
                precision = len(TP) / len(high_pred)

            recall = len(TP) / len(high_gt)
            P.append(precision)
            R.append(recall)

            # accuracy
            TPTN = (len(TP) + len(low_gt & low_pred))
            accuracy = TPTN / test_triples.shape[0]
            Acc.append(accuracy)

        P = np.array(P)
        R = np.array(R)
        F1 = 2 * np.multiply(P, R) / (P + R)
        Acc = np.array(Acc)

        return scores, P, R, F1, Acc


    def decision_tree_classify(self, confT, train_data):
        """
        :param confT: :param confT: the threshold of ground truth confidence score
        :param train_data: dataframe['v1','relation','v2','w']
        :return:
        """
        # train_data = pd.read_csv(os.path.join(data_dir,'train.tsv'), sep='\t', header=None, names=['v1','relation','v2','w'])

        test_triples = self.test_triples

        # train
        train_h, train_r, train_t = train_data['v1'].values.astype(int), train_data['relation'].values.astype(int), train_data['v2'].values.astype(int)
        train_X = self.get_score_batch(train_h, train_r, train_t)[:, np.newaxis]  # feature(2D, n*1)
        train_Y = train_data['w']>confT  # label (high confidence/not)
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_X, train_Y)

        # predict
        test_triples = self.test_triples
        test_h, test_r, test_t = test_triples[:, 0].astype(int), test_triples[:, 1].astype(int), test_triples[:, 2].astype(int)
        test_X = self.get_score_batch(test_h, test_r, test_t)[:, np.newaxis]
        test_Y_truth = test_triples[:, 3]>confT
        test_Y_pred = clf.predict(test_X)
        print('Number of true positive: %d' % np.sum(test_Y_truth))
        print('Number of predicted positive: %d'%np.sum(test_Y_pred))


        precision, recall, F1, _ = sklearn.metrics.precision_recall_fscore_support(test_Y_truth, test_Y_pred)
        accu = sklearn.metrics.accuracy_score(test_Y_truth, test_Y_pred)

        # P-R curve
        P, R, thres = sklearn.metrics.precision_recall_curve(test_Y_truth, test_X)

        return test_X, precision, recall, F1, accu, P, R

    def get_fixed_hr(self, outputdir=None, n=500):
        hr_map500 = {}
        dict_keys = []
        for h in self.hr_map.keys():
            for r in self.hr_map[h].keys():
                dict_keys.append([h, r])

        dict_keys = sorted(dict_keys, key=lambda x: len(self.hr_map[x[0]][x[1]]), reverse=True)
        dict_final_keys = []

        for i in range(2525):
            dict_final_keys.append(dict_keys[i])

        count = 0
        for i in range(n):
            temp_key = random.choice(dict_final_keys)
            h = temp_key[0]
            r = temp_key[1]
            for t in self.hr_map[h][r]:
                w = self.hr_map[h][r][t]
                if hr_map500.get(h) == None:
                    hr_map500[h] = {}
                if hr_map500[h].get(r) == None:
                    hr_map500[h][r] = {t: w}
                else:
                    hr_map500[h][r][t] = w

        for h in hr_map500.keys():
            for r in hr_map500[h].keys():
                count = count + 1

        self.hr_map_sub = hr_map500

        if outputdir is not None:
            with open(outputdir, 'wb') as handle:
                pickle.dump(hr_map500, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return hr_map500







class U_ANALOGY_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)

    def build_by_var(self, test_data, epoch, this_data, dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
      
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c = parameter_list['ent_embedding.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_embedding.weight'].cpu().detach().numpy()
        self.vec_c_real = parameter_list['ent_embeddings_real.weight'].cpu().detach().numpy()
        self.vec_r_real = parameter_list['rel_embeddings_real.weight'].cpu().detach().numpy()
        self.vec_c_img = parameter_list['ent_embeddings_img.weight'].cpu().detach().numpy()
        self.vec_r_img = parameter_list['rel_embeddings_img.weight'].cpu().detach().numpy()
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        h, hvec, hvec_img, r, rvec, rvec_img, t, tvec, tvec_img = self.vecs_from_triples(h, r, t)
        if param.function == 'logi':
            return sigmoid(self.w * (np.sum(np.multiply(np.multiply(hvec, tvec), rvec) +
                                   np.multiply(np.multiply(hvec_img, tvec_img), rvec) +
                                   np.multiply(np.multiply(hvec, tvec_img), rvec_img) -
                                   np.multiply(np.multiply(hvec_img, tvec), rvec_img)) +
                                   np.sum(np.multiply(np.multiply(h, t), r))) + self.b)
        return self.w * (np.sum(np.multiply(np.multiply(hvec, tvec), rvec) +
                                   np.multiply(np.multiply(hvec_img, tvec_img), rvec) +
                                   np.multiply(np.multiply(hvec, tvec_img), rvec_img) -
                                   np.multiply(np.multiply(hvec_img, tvec), rvec_img)) +
                                   np.sum(np.multiply(np.multiply(h, t), r))) + self.b

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        h, hvec, hvec_img = self.con_index2vec_batch(h_batch)
        r, rvec, rvec_img = self.rel_index2vec_batch(r_batch)
        t, tvec, tvec_img = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * (np.sum(np.multiply(np.multiply(hvec, tvec), rvec) +
                               np.multiply(np.multiply(hvec_img, tvec_img), rvec) +
                               np.multiply(np.multiply(hvec, tvec_img), rvec_img) -
                               np.multiply(np.multiply(hvec_img, tvec), rvec_img), axis=axis) +
                         np.sum(np.multiply(np.multiply(h, t), r), axis=axis)) + self.b)
        return self.w * (np.sum(np.multiply(np.multiply(hvec, tvec), rvec) +
                               np.multiply(np.multiply(hvec_img, tvec_img), rvec) +
                               np.multiply(np.multiply(hvec, tvec_img), rvec_img) -
                               np.multiply(np.multiply(hvec_img, tvec), rvec_img), axis=axis) +
                         np.sum(np.multiply(np.multiply(h, t), r), axis=axis)) + self.b


    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)


    def con_index2vec(self, c):
        return self.vec_c[c], self.vec_c_real[c], self.vec_c_img[c]

    def rel_index2vec(self, r):
        return self.vec_r[r], self.vec_r_real[r], self.vec_r_img[r]
    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        h, hvec, hvec_img = self.con_index2vec(h)
        r, rvec, rvec_img = self.rel_index2vec(r)
        t, tvec, tvec_img = self.con_index2vec(t)
        return h, hvec, hvec_img, r, rvec, rvec_img, t, tvec, tvec_img
    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :]), np.squeeze(self.vec_c_real[[indices], :]), np.squeeze(self.vec_c_img[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :]), np.squeeze(self.vec_r_real[[indices], :]), np.squeeze(self.vec_r_img[[indices], :])



class U_ComplEx_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)

    def load_checkpoint(self, path):
        res = []
        a = torch.load(os.path.join(path))
        # self.load_state_dict(a)
        for _, v in a.items():
            # print(v)
            v = v.cpu().detach().numpy()
            res.append(v)
        return res
    def build_by_var(self, test_data, epoch, this_data, dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c_real = parameter_list['ent_embeddings_real.weight'].cpu().detach().numpy()
        self.vec_r_real = parameter_list['rel_embeddings_real.weight'].cpu().detach().numpy()
        self.vec_c_img = parameter_list['ent_embeddings_img.weight'].cpu().detach().numpy()
        self.vec_r_img = parameter_list['rel_embeddings_img.weight'].cpu().detach().numpy()
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        hvec, hvec_img, rvec, rvec_img, tvec, tvec_img = self.vecs_from_triples(h, r, t)
        if param.function == 'logi':
            return sigmoid(self.w * np.sum(np.multiply(np.multiply(hvec, tvec), rvec) +
                               np.multiply(np.multiply(hvec_img, tvec_img), rvec) +
                               np.multiply(np.multiply(hvec, tvec_img), rvec_img) -
                               np.multiply(np.multiply(hvec_img, tvec), rvec_img)) + self.b)
        return self.w * np.sum(np.multiply(np.multiply(hvec, tvec), rvec) +
                               np.multiply(np.multiply(hvec_img, tvec_img), rvec) +
                               np.multiply(np.multiply(hvec, tvec_img), rvec_img) -
                               np.multiply(np.multiply(hvec_img, tvec), rvec_img)) + self.b

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        hvec, hvec_img = self.con_index2vec_batch(h_batch)
        rvec, rvec_img = self.rel_index2vec_batch(r_batch)
        tvec, tvec_img = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * np.sum(np.multiply(np.multiply(hvec, tvec), rvec) +
                               np.multiply(np.multiply(hvec_img, tvec_img), rvec) +
                               np.multiply(np.multiply(hvec, tvec_img), rvec_img) -
                               np.multiply(np.multiply(hvec_img, tvec), rvec_img), axis=axis) + self.b)
        return self.w * np.sum(np.multiply(np.multiply(hvec, tvec), rvec) +
                               np.multiply(np.multiply(hvec_img, tvec_img), rvec) +
                               np.multiply(np.multiply(hvec, tvec_img), rvec_img) -
                               np.multiply(np.multiply(hvec_img, tvec), rvec_img), axis=axis) + self.b


    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)

    def con_index2vec(self, c):
        return self.vec_c_real[c], self.vec_c_img[c]

    def rel_index2vec(self, r):
        return self.vec_r_real[r], self.vec_r_img[r]
    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c_real[[indices], :]), np.squeeze(self.vec_c_img[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r_real[[indices], :]), np.squeeze(self.vec_r_img[[indices], :])
    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec, hvec_img = self.con_index2vec(h)
        rvec, rvec_img = self.rel_index2vec(r)
        tvec, tvec_img = self.con_index2vec(t)
        return hvec, hvec_img, rvec, rvec_img, tvec, tvec_img

class U_SLM_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)
    def load_checkpoint(self, path):
        res = []
        a = torch.load(os.path.join(path))
        # self.load_state_dict(a)
        for _, v in a.items():
            # print(v)
            v = v.cpu().detach().numpy()
            res.append(v)
        return res
    def build_by_var(self, test_data, epoch, this_data, dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c = parameter_list['ent_embedding.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_embedding.weight'].cpu().detach().numpy()
        self.vec_mr1 = parameter_list['mr1.weight'].cpu().detach().numpy()
        self.vec_mr2 = parameter_list['mr2.weight'].cpu().detach().numpy()
        # self.vec_c_img = parameter_list['ent_embeddings_img.weight'].cpu().detach().numpy()
        # self.vec_r_img = parameter_list['rel_embeddings_img.weight'].cpu().detach().numpy()
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0
    def layer(self, h, t):
        """Defines the forward pass layer of the algorithm.

          Args:
              h (Tensor): Head entities ids.
              t (Tensor): Tail entity ids of the triple.
        """
        mr1h = np.matmul(h, self.vec_mr1) # h => [m, d], self.mr1 => [d, k]
        mr2t = np.matmul(t, self.vec_mr2) # t => [m, d], self.mr2 => [d, k]
        return np.tanh(mr1h + mr2t)

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        h, r, t = self.vecs_from_triples(h, r, t)
        if param.function == 'logi':
            return sigmoid(self.w * (np.sum(np.multiply(self.layer(h, t), r))) + self.b)
        return self.w * (np.sum(np.multiply(self.layer(h, t), r))) + self.b

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        h = self.con_index2vec_batch(h_batch)
        r = self.rel_index2vec_batch(r_batch)
        t = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * (np.sum(np.multiply(self.layer(h, t), r), axis=axis)) + self.b)
        return self.w * (np.sum(np.multiply(self.layer(h, t), r), axis=axis)) + self.b


    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)


    def con_index2vec(self, c):
        return self.vec_c[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]
    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :])
    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        h = self.con_index2vec(h)
        r = self.rel_index2vec(r)
        t = self.con_index2vec(t)
        return h, r, t

class U_RESCAL_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)

    # override
    def load_checkpoint(self, path):
        res = []
        a = torch.load(os.path.join(path))
        # self.load_state_dict(a)
        for _, v in a.items():
            # print(v)
            v = v.cpu().detach().numpy()
            res.append(v)
        return res
    def build_by_var(self, test_data, epoch, this_data ,dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c = parameter_list['ent_embeddings.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_matrices.weight'].cpu().detach().numpy()
        # self.vec_t = parameter_list['obj_embeddings.weight'].cpu().detach().numpy()
        # a = res[2]
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0
    def _calc(self, h, t, r):
        t = np.expand_dims(t, -1)
        r = r.reshape(128, 128)
        tr = np.matmul(r, t)
        tr = tr.squeeze()
        return np.sum(h * tr, -1)
    def _calc2(self, h, t, r, dim):
        t = np.expand_dims(t, -1)
        if dim == 1:
            r = r.reshape(-1, 128, 128)
        elif dim == 2:
            r = r.reshape(-1, 10, 128, 128)
        tr = np.matmul(r, t)
        tr = tr.squeeze()
        return np.sum(h * tr, -1)
    # override
    def get_score(self, h, r, t):
        # no sigmoid
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        score = self._calc(hvec, tvec, rvec)
        a = self.w * score + self.b
        if param.function == 'logi':
            return sigmoid(a)

        return a

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)

        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum

        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * self._calc2(hvecs, tvecs, rvecs, axis) + self.b)
        return self.w * self._calc2(hvecs, tvecs, rvecs, axis) + self.b




    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)

    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec = self.con_index2vec(h)
        rvec= self.rel_index2vec(r)
        tvec = self.con_index2vec(t)
        return hvec, rvec, tvec
    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :])
    def con_index2vec(self, c):
        return self.vec_c[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]



class U_CP_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)


    # override
    def load_checkpoint(self, path):
        res = []
        a = torch.load(os.path.join(path))
        # self.load_state_dict(a)
        for _, v in a.items():
            # print(v)
            v = v.cpu().detach().numpy()
            res.append(v)
        return res
    def build_by_var(self, test_data, epoch, this_data ,dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_h = parameter_list['sub_embeddings.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_embeddings.weight'].cpu().detach().numpy()
        self.vec_t = parameter_list['obj_embeddings.weight'].cpu().detach().numpy()
        # a = res[2]
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        a = self.w * np.sum(rvec * (hvec * tvec)) + self.b
        if param.function == 'logi':
            return sigmoid(a)

        return a

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.tail_index2vec_batch(t_batch)

        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum

        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * np.sum(rvecs * (hvecs * tvecs), axis=axis) + self.b)
        return self.w * np.sum(rvecs * (hvecs * tvecs), axis=axis) + self.b




    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)

    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec = self.con_index2vec(h)
        rvec= self.rel_index2vec(r)
        tvec = self.tail_index2vec(t)
        return hvec, rvec, tvec
    def con_index2vec(self, c):
        return self.vec_h[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]
    def tail_index2vec(self, c):
        return self.vec_t[c]
    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_h[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :])
    def tail_index2vec_batch(self, indices):
        return np.squeeze(self.vec_t[[indices], :])


class U_RotatE_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)


    # override
    def load_checkpoint(self, path):
        res = []
        a = torch.load(os.path.join(path))
        # self.load_state_dict(a)
        for _, v in a.items():
            # print(v)
            v = v.cpu().detach().numpy()
            res.append(v)
        return res
    def build_by_var(self, test_data, epoch, this_data, dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c = parameter_list['ent_embedding.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_embedding.weight'].cpu().detach().numpy()
        # self.vec_mr1 = parameter_list['mr1.weight'].cpu().detach().numpy()
        # self.vec_mr2 = parameter_list['mr2.weight'].cpu().detach().numpy()
        self.vec_c_img = parameter_list['ent_embeddings_imag.weight'].cpu().detach().numpy()
        # self.vec_r_img = parameter_list['rel_embeddings_img.weight'].cpu().detach().numpy()
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0


    # override
    def get_score(self, h, r, t):
        # no sigmoid
        pi = 3.14159265358979323846
        h, h_img, r, t, t_img = self.vecs_from_triples(h, r, t)
        r = r / ((2.0+2.0)/128.0 / pi)
        r_img = np.sin(r)
        r = np.cos(r)
        if param.function == 'logi':
            return sigmoid(self.w * (2.0- np.sum((h*r - h_img*r_img - t)**2 + (h*r_img + h_img*r -t_img)**2)) + self.b)
        return self.w * (2.0- np.sum((h*r - h_img*r_img - t)**2 + (h*r_img + h_img*r -t_img)**2)) + self.b

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        pi = 3.14159265358979323846
        h, h_img = self.con_index2vec_batch(h_batch)
        r = self.rel_index2vec_batch(r_batch)
        t, t_img = self.con_index2vec_batch(t_batch)
        r = r / ((2.0+2.0)/128.0 / pi)
        r_img = np.sin(r)
        r = np.cos(r)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * (2.0 - np.sum((h*r - h_img*r_img - t)**2 + (h*r_img + h_img*r -t_img)**2, axis=axis)) + self.b)
        return self.w * (2.0 - np.sum((h*r - h_img*r_img - t)**2 + (h*r_img + h_img*r -t_img)**2, axis=axis)) + self.b


    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)


    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        h, h_img = self.con_index2vec(h)
        r = self.rel_index2vec(r)
        t, t_img = self.con_index2vec(t)
        return h, h_img, r, t, t_img
    def con_index2vec(self, c):
        return self.vec_c[c], self.vec_c_img[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]
    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :]),np.squeeze(self.vec_c_img[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :])


class U_SimplE_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)


    def build_by_var(self, test_data, epoch, this_data , dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c_h = parameter_list['ent_head_embeddings.weight'].cpu().detach().numpy()
        self.vec_c_t = parameter_list['ent_tail_embeddings.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_embeddings.weight'].cpu().detach().numpy()
        self.vec_r_inv = parameter_list['rel_inv_embeddings.weight'].cpu().detach().numpy()
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        h1_e, h2_e, r1_e, r2_e, t1_e, t2_e = self.vecs_from_triples(h, r, t)
        if param.function == 'logi':
            return sigmoid(self.w * (np.sum(h1_e*r1_e*t1_e) + np.sum(h2_e*r2_e*t2_e) / 2.0) + self.b)
        return self.w * (np.sum(h1_e*r1_e*t1_e) + np.sum(h2_e*r2_e*t2_e) / 2.0) + self.b

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        h1, t2 = self.con_index2vec_batch(h_batch)
        r1, r2 = self.rel_index2vec_batch(r_batch)
        h2, t1 = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * (np.sum(h1*r1*t1, axis) + np.sum(h2*r2*t2, axis) / 2.0) + self.b)
        return self.w * (np.sum(h1*r1*t1, axis) + np.sum(h2*r2*t2, axis) / 2.0) + self.b


    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)


    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        h1, t2 = self.con_index2vec(h)
        r1, r2 = self.rel_index2vec(r)
        h2, t1 = self.con_index2vec(t)
        return h1, h2, r1, r2, t1, t2
    def con_index2vec(self, c):
        return self.vec_c_h[c], self.vec_c_t[c]

    def rel_index2vec(self, r):
        return self.vec_r[r], self.vec_r_inv[r]

    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c_h[[indices], :]), np.squeeze(self.vec_c_t[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :]), np.squeeze(self.vec_r_inv[[indices], :])


class UH_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)


    def build_by_var(self, test_data, epoch, this_data ,dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c = parameter_list['ent_embedding.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_embedding.weight'].cpu().detach().numpy()
        self.vec_w = parameter_list['w.weight'].cpu().detach().numpy()
        # a = res[2]
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
      
        a = self.w * np.sum((hvec * rvec) * tvec) + self.b
        if param.function == 'logi':
            return sigmoid(a)
        return a

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs, wvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)
        hvecs = self._projection(hvecs, wvecs)
        tvecs = self._projection(tvecs, wvecs)
  
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
 
        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * np.sum((hvecs * rvecs) * tvecs, axis=axis) + self.b)
        return self.w * np.sum((hvecs * rvecs) * tvecs, axis=axis) + self.b




    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)


    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec = self.con_index2vec(h)
        rvec, wvec = self.rel_index2vec(r)
        tvec = self.con_index2vec(t)
        hvec = self._projection(hvec, wvec)
        tvec = self._projection(tvec, wvec)
        return hvec, rvec, tvec
    @staticmethod
    def _projection(emb_e, proj_vec):
        """Calculates the projection of entities"""
        proj_vec = proj_vec/np.linalg.norm(proj_vec, ord=2, axis=-1, keepdims=True)

        # [b, k], [b, k]
        return emb_e - np.sum(emb_e * proj_vec, axis=-1, keepdims=True) * proj_vec
    def con_index2vec(self, c):
        return self.vec_c[c]

    def rel_index2vec(self, r):
        return self.vec_r[r],self.vec_w[r]
    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :]), np.squeeze(self.vec_w[[indices], :])
    


class UD_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)

    # override

    def build_by_var(self, test_data, epoch, this_data, dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c = parameter_list['ent_embedding.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_embedding.weight'].cpu().detach().numpy()
        self.vec_cm = parameter_list['ent_mappings.weight'].cpu().detach().numpy()
        self.vec_rm = parameter_list['ent_mappings.weight'].cpu().detach().numpy()
        # a = res[2]
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)

        a = self.w * np.sum((hvec * tvec) * rvec) + self.b
        if param.function == 'logi':
            return sigmoid(a)
        return a

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        hvecs, hmvecs = self.con_index2vec_batch(h_batch)
        rvecs, rmvecs = self.rel_index2vec_batch(r_batch)
        tvecs, tmvecs = self.con_index2vec_batch(t_batch)
        hvecs = self._projection(hvecs, hmvecs, rmvecs)
        tvecs = self._projection(tvecs, tmvecs, rmvecs)

        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum

        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * np.sum((hvecs * tvecs) * rvecs, axis=axis) + self.b)
        return self.w * np.sum((hvecs * tvecs) * rvecs, axis=axis) + self.b




    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)


    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec,hvecm = self.con_index2vec(h)
        rvec, rvecm = self.rel_index2vec(r)
        tvec, tvecm = self.con_index2vec(t)
        hvec = self._projection(hvec, hvecm, rvecm)
        tvec = self._projection(tvec, tvecm, rvecm)
        return hvec, rvec, tvec

    @staticmethod
    def _projection(emb_e, emb_m, proj_vec):
        """Calculates the projection of entities"""
        # [b, k] + sigma ([b, k] * [b, k]) * [b, k]
        return emb_e + np.sum(emb_e * emb_m, axis=-1, keepdims=True) * proj_vec
    def con_index2vec(self, c):
        return self.vec_c[c], self.vec_cm[c]

    def rel_index2vec(self, r):
        return self.vec_r[r], self.vec_rm[r]

    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :]), np.squeeze(self.vec_cm[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :]), np.squeeze(self.vec_rm[[indices], :])

    
class UKGE_Tester(Tester):
    def __init__(self, modelname, num_rels, num_cons):
        Tester.__init__(self, modelname, num_rels, num_cons)

  

    # override
    def load_checkpoint(self, path):
        res = []
        a = torch.load(os.path.join(path))
        # self.load_state_dict(a)
        for _, v in a.items():
            # print(v)
            v = v.cpu().detach().numpy()
            res.append(v)
        return res
    def build_by_var(self, test_data, epoch, this_data ,dir):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        # self.tf_parts = tf_model

        # value_ht, value_r, w, b = sess.run(
        #     [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        parameter_list = torch.load(os.path.join(dir + str(epoch) + 'KE.ckpt'))
        self.vec_c = parameter_list['ent_embedding.weight'].cpu().detach().numpy()
        self.vec_r = parameter_list['rel_embedding.weight'].cpu().detach().numpy()
        # self.vec_w = parameter_list['w.weight'].cpu().detach().numpy()
        # a = res[2]
        self.w = float(parameter_list['liner.weight'])
        self.b = float(parameter_list['liner.bias'])
        return 0

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)

        a = self.w * np.sum((hvec * tvec) * rvec) + self.b
        if param.function == 'logi':
            return sigmoid(a)
        return a

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)


        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum

        else:
            axis = 1
        if param.function == 'logi':
            return sigmoid(self.w * np.sum((hvecs * tvecs) * rvecs, axis=axis) + self.b)
        return self.w * np.sum((hvecs * tvecs) * rvecs, axis=axis) + self.b




    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)



    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec = self.con_index2vec(h)
        rvec= self.rel_index2vec(r)
        tvec = self.con_index2vec(t)
        return hvec, rvec, tvec

    def con_index2vec(self, c):
        return self.vec_c[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]

    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :])
