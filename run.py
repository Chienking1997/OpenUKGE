from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if './src' not in sys.path:
    sys.path.append('./src')

if './' not in sys.path:
    sys.path.append('./')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from os.path import join
from src.data import Data

from src.trainer import Trainer
from src.list import ModelList
import datetime

import argparse
from src import param


def get_model_identifier(whichmodel, function):
    prefix = whichmodel.value
    now = datetime.datetime.now()
    date = '%02d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
    identifier = prefix + '_' + function +'_'+ date
    return identifier


parser = argparse.ArgumentParser()
# required
parser.add_argument('--data', type=str, default='nl27k',
                    help="the dir path where you store data (train.tsv, val.tsv, test.tsv). Default: ppi5k")
# optional
parser.add_argument("--verbose", help="print detailed info for debugging",
                    action="store_true")
parser.add_argument('-m', '--model', type=str, default='ukge', help="choose model . default: uanalogy")
parser.add_argument('-f', '--function', type=str, default='logi', help="choose maping function rect or logi. default: rect")
parser.add_argument('-d', '--dim', type=int, default=128, help="set dimension. default: 128")
parser.add_argument('--epoch', type=int, default=1000, help="set number of epochs. default: 100")
parser.add_argument('--lr', type=float, default=0.001, help="set learning rate. default: 0.001")
parser.add_argument('--batch_size', type=int, default=1024, help="set batch size. default: 1024")
parser.add_argument('--n_neg', type=int, default=10, help="Number of negative samples per (h,r,t). default: 10")
parser.add_argument('--save_freq', type=int, default=10,
                    help="how often (how many epochs) to run validation and save tf models. default: 50")
parser.add_argument('--models_dir', type=str, default='./trained_models',
                    help="the dir path where you store trained models. A new directory will be created inside it.")

# regularizer coefficient (lambda)
parser.add_argument('--reg_scale', type=float, default=0.0001,
                    help="The scale for regularizer (lambda). Default 0.0005")

args = parser.parse_args()

# parameters
param.verbose = args.verbose
param.whichdata = args.data
param.model = args.model
param.function = args.function
param.whichmodel = ModelList(args.model)
param.n_epoch = args.epoch
param.learning_rate = args.lr
param.batch_size = args.batch_size
param.val_save_freq = args.save_freq  # The frequency to validate and save model
param.dim = args.dim  # default 128
param.neg_per_pos = args.n_neg  # Number of negative samples per (h,r,t). default 10.
param.reg_scale = args.reg_scale

# path to save
identifier = get_model_identifier(param.whichmodel, param.function)
save_dir = join(args.models_dir, param.whichdata, identifier)  # the directory where we store this model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('Trained models will be stored in: ', save_dir)

# input files
data_dir = join('./data', args.data)
file_train = join(data_dir, 'train.tsv')  # training data
file_val = join(data_dir, 'val.tsv')  # validation datan
file_psl = join(data_dir, 'softlogic.tsv')  # probabilistic soft logic
file_test = join(data_dir, 'test.tsv')
print('file_psl: %s' % file_psl)

more_filt = [file_val, join(data_dir, 'test.tsv')]
print('Read train.tsv from', data_dir)

# load data
this_data = Data()
this_data.load_data(file_train=file_train, file_val=file_val, file_test=file_test, file_psl=file_psl)
for f in more_filt:
    this_data.record_more_data(f)
this_data.save_meta_table(save_dir)  # output: idx_concept.csv, idx_relation.csv

m_train = Trainer()
m_train.build(this_data, save_dir, lr=param.learning_rate, modelname=param.model)

# Model will be trained, validated, and saved in './trained_models'
m_train.train(epochs=param.n_epoch, save_every_epoch=param.val_save_freq,
                                          lr=param.learning_rate,
                                          data_dir=param.data_dir())
