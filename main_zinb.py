import torch
from torch import optim
from torch.nn import functional as F

import math
import argparse
import os
import sys
import time
# import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pickle

sys.path.insert(1, '../_model_source_codes/')

from GMVAE_model_zinb import GMVAE_ZINB
from GMVAE_losses_zinb import *
from GMVAE_utils import train as gmvae_train
from GMVAE_utils import test as gmvae_test

parser = argparse.ArgumentParser(description='Gaussian Mixture VAE with ZINB')
parser.add_argument('-dl','--data-loc', type=str, default="./data_processed/lupus_gmvae/", help='data location')
parser.add_argument('-tdl','--testdata-loc', type=str, default="./data_processed/lupus_gmvae/", help='testdata location')
parser.add_argument('-msl','--modelsaveloc', type=str, default="models", help='model save location')
args, unknown = parser.parse_known_args()

# model folder is where trained models are saved
if not (os.path.isdir(args.modelsaveloc)):
    os.mkdir(args.modelsaveloc)

# args.data_loc must have one mtx file and one csv file
#### train data loading ####
for f in os.listdir(args.data_loc):
	if (('.mtx' in f) or ('.mtx.gz' in f)):
		raw_count_file=f
	if (('.csv' in f) or ('.csv.gz' in f)):
		cluster_file=f

print(f'Loading data from {args.data_loc}')
print(f'raw count file is {raw_count_file}')

scRNA_data = scipy.io.mmread(args.data_loc + raw_count_file)
scRNA_data_array = scRNA_data.toarray()
scRNA_data_array = scRNA_data_array.astype('float32')
scRNA_data_array = torch.tensor(scRNA_data_array)

print(f'clustering file is {cluster_file}')
cluster = np.genfromtxt(args.data_loc + cluster_file, delimiter=",", dtype=int, skip_header=True)

cluster = torch.tensor(cluster)

scRNA_data_set = TensorDataset(scRNA_data_array.T, cluster)

print(type(scRNA_data_array))
print(f'scRNA data shape {scRNA_data_array.shape}')

print(type(cluster))
print(f'cluster data shape {cluster.shape}')

print(f'Number of clusters {len(torch.unique(cluster))}')
print(f'Number of genes {scRNA_data_array.size(0)}')
#### train data loaded ####


# args.testdata_loc must have one mtx file and one csv file
#### test data loading ####
for f in os.listdir(args.testdata_loc):
	if (('.mtx' in f) or ('.mtx.gz' in f)):
		raw_count_file=f
	if (('.csv' in f) or ('.csv.gz' in f)):
		cluster_file=f

print(f'Loading data from {args.testdata_loc}')
print(f'raw count file is {raw_count_file}')

scRNA_data = scipy.io.mmread(args.testdata_loc + raw_count_file)
scRNA_data_array = scRNA_data.toarray()
scRNA_data_array = scRNA_data_array.astype('float32')
scRNA_data_array = torch.tensor(scRNA_data_array)

print(f'clustering file is {cluster_file}')
cluster = np.genfromtxt(args.testdata_loc + cluster_file, delimiter=",", dtype=int, skip_header=True)

cluster = torch.tensor(cluster)

scRNA_testdata_set = TensorDataset(scRNA_data_array.T, cluster)

print(type(scRNA_data_array))
print(f'scRNA data shape {scRNA_data_array.shape}')

print(type(cluster))
print(f'cluster data shape {cluster.shape}')

print(f'Number of clusters {len(torch.unique(cluster))}')
print(f'Number of genes {scRNA_data_array.size(0)}')

#### train data loaded ####
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
					help='number of epochs to train (default: 13)')
parser.add_argument('--epochs-restart', type=int, default=1000000, metavar='N',
					help='restart epoch')
parser.add_argument('--seed', type=int, default=123, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--K', type=int, metavar='N',
					help='number of clusters')
parser.add_argument('--input_dim', type=int, default=784, metavar='N',
					help='dimension of input')
parser.add_argument('--h_dim', type=int, default=128, metavar='N',
					help='dimension of hidden layer')
parser.add_argument('--h_dim1', type=int, default=128, metavar='N',
					help='dimension of hidden layer')
parser.add_argument('--h_dim2', type=int, default=64, metavar='N',
					help='dimension of hidden layer')
parser.add_argument('--z_dim', type=int, default=32, metavar='N',
					help='dimension of hidden layer')
parser.add_argument('--savefreq', type=int, default=50, metavar='N', help='model save frequency')
parser.add_argument('--testfreq', type=int, default=50, metavar='N', help='model save frequency')
parser.add_argument('--dataset', default='lupus', help='dataset to use')
parser.add_argument('-lr','--learning-rate', type=float, default=6e-6,
					help='learning rate for optimizer')
parser.add_argument('-rs','--restart', type=bool, default=False, help='restarting')
parser.add_argument('-mf','--model-file', type=str, default="saved_model.pt", help='saved model to load')
parser.add_argument('-mn','--model-name', type=str, default="GMVAE_w_zinb", help='model description')

args, unknown = parser.parse_known_args()
args.input_dim = scRNA_data_array.size(0)


args.K = len(torch.unique(cluster))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
torch.manual_seed(args.seed)

#Data loader
print(f'Data loader with the batch size = {args.batch_size}')
train_loader = DataLoader(scRNA_data_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(scRNA_testdata_set, batch_size=args.batch_size, shuffle=True)

gmvae = GMVAE_ZINB(args).to(device)
print(gmvae)
print(vars(args))

with open("args_save_zinb.pickle", "wb") as output_file:
    pickle.dump(args, output_file)

if args.restart:
    args.model_file = args.model_file

    gmvae.load_state_dict(torch.load(args.model_file, weights_only=True))
    gmvae.train()                             
    
if not (args.restart):
    print(f'Training from 1 to {args.epochs}')
    time_stamp = time.ctime()[4:24].replace("  ","_").replace(" ","_").replace(":","_")
    tf_writer = SummaryWriter(filename_suffix=args.model_name, comment=f"_started_{time_stamp}")
    optimizer = optim.Adam(gmvae.parameters(), lr=args.learning_rate)

    progress_bar = tqdm(range(1, args.epochs), unit = " epoch")
    for epoch in progress_bar:
        # train the network
        total_loss_epoch, KLD_gaussian_epoch, KLD_pi_epoch, zinb_loss_epoch, accuracy \
         = gmvae_train(epoch, gmvae, train_loader, optimizer)
        # print(zinb_loss_epoch)
        tf_writer.add_scalar("total_loss/train", total_loss_epoch, epoch)
        tf_writer.add_scalar("KLD_gaussian/train", KLD_gaussian_epoch, epoch)
        tf_writer.add_scalar("KLD_pi/train", KLD_pi_epoch, epoch)
        tf_writer.add_scalar("reconst_loss/train", zinb_loss_epoch, epoch)
        tf_writer.add_scalar("accuracy/train", accuracy, epoch)

        if epoch % args.testfreq == 0:
            total_loss_test, KLD_gaussian_test, KLD_pi_test, zinb_loss_test, accuracy_test \
            = gmvae_test(epoch, gmvae, test_loader, optimizer)

            tf_writer.add_scalar("total_loss/test", total_loss_test, epoch)
            tf_writer.add_scalar("KLD_gaussian/test", KLD_gaussian_test, epoch)
            tf_writer.add_scalar("KLD_pi/test", KLD_pi_test, epoch)
            tf_writer.add_scalar("reconst_loss/test", zinb_loss_test, epoch)
            tf_writer.add_scalar("accuracy/test", accuracy_test, epoch)
 
        if epoch % args.savefreq == 0:
            model_file = f'./{args.modelsaveloc}/{args.dataset}_zdim_{args.z_dim}_bs_{args.batch_size}_{epoch}_{args.model_name}_wholePTmodel.pt'
            torch.save(gmvae, model_file)
        
if args.restart:
    print(f'Training restart from {args.epochs_restart} to {args.epochs}')
    time_stamp = time.ctime()[4:24].replace("  ","_").replace(" ","_").replace(":","_")
    tf_writer = SummaryWriter(filename_suffix=args.model_name, comment=f"_restart_from_{args.epochs_restart}_{time_stamp}")
    optimizer = optim.Adam(gmvae.parameters(), lr=args.learning_rate)
    
    if args.epochs_restart >= args.epochs:
        print(f"Epochs should be more than the restarting point")
    
    progress_bar = tqdm(range(args.epochs_restart, args.epochs), unit = " epoch")
    for epoch in progress_bar:
        # train the network
        total_loss_epoch, KLD_gaussian_epoch, KLD_pi_epoch, zinb_loss_epoch, accuracy \
         = gmvae_train(epoch, gmvae, train_loader, optimizer)
        # 
        tf_writer.add_scalar("total_loss/train", total_loss_epoch, epoch)
        tf_writer.add_scalar("KLD_gaussian/train", KLD_gaussian_epoch, epoch)
        tf_writer.add_scalar("KLD_pi/train", KLD_pi_epoch, epoch)
        tf_writer.add_scalar("reconst_loss/train", zinb_loss_epoch, epoch)
        tf_writer.add_scalar("accuracy/train", accuracy, epoch)

        #
        if epoch % args.testfreq == 0:
            total_loss_test, KLD_gaussian_test, KLD_pi_test, zinb_loss_test, accuracy_test \
            = gmvae_test(epoch, gmvae, test_loader, optimizer)

            tf_writer.add_scalar("total_loss/test", total_loss_test, epoch)
            tf_writer.add_scalar("KLD_gaussian/test", KLD_gaussian_test, epoch)
            tf_writer.add_scalar("KLD_pi/test", KLD_pi_test, epoch)
            tf_writer.add_scalar("reconst_loss/test", zinb_loss_test, epoch)
            tf_writer.add_scalar("accuracy/test", accuracy_test, epoch)
            
        if epoch % args.savefreq == 0:
            model_file = f'./{args.modelsaveloc}/{args.dataset}_zdim_{args.z_dim}_bs_{args.batch_size}_{epoch}_{args.model_name}_wholePTmodel.pt'
            torch.save(gmvae, model_file)
            # torch.save(gmvae.state_dict(), model_file)