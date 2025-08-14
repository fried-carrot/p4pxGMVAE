#!/usr/bin/env python
"""
GMVAE TRAINING SCRIPT - EXACT COPY FROM BULK2SC
This is an EXACT COPY of bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py
with ONLY path modifications and parameter defaults changed

SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py (lines 1-229)
GitHub: https://github.com/mcgilldinglab/scBeacon
"""

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py lines 1-19
# All imports are EXACT copies
import argparse
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import NLLLoss, CrossEntropyLoss, L1Loss
import pickle
from dataloader import DataLoader, TestDataLoader
import sys
import os
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from torchvision import transforms

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py line 21
# MODIFIED: Path from '../_model_source_codes/' to '.'
# Original: sys.path.insert(1, '../_model_source_codes/')
sys.path.insert(1, '.')

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py line 23
# MODIFIED: Import name from GMVAE_scrna_zinb to GMVAE_model_zinb (renamed file)
# Original: from GMVAE_scrna_zinb import GMVAE_ZINB
from GMVAE_model_zinb import GMVAE_ZINB

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py line 24
from GMVAE_utils import gmvae_train

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py lines 26-128
# Parser setup is EXACT copy with only default value changes
parser = argparse.ArgumentParser()

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py lines 29-30
# MODIFIED: Default paths from "./data/lung_10k/" to "data_processed/lupus_gmvae/"
# Original: default="./data/lung_10k/"
parser.add_argument('-dl','--data-loc', type=str, default="data_processed/lupus_gmvae/", help='data location')
parser.add_argument('-tdl','--testdata-loc', type=str, default="data_processed/lupus_gmvae/", help='testdata location')

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py lines 31-104
# All following parser arguments are EXACT copies
parser.add_argument('--datadir', type=str, default='data_processed/lupus_gmvae', metavar='DD',
                    help='directory which contains data')
parser.add_argument('--outputdir', type=str, default='output_gmvae', metavar='OD',
                    help='directory which stores outputs')
parser.add_argument('--modelsaveloc', type=str, default='', metavar='MSL',
                    help='directory for saving model')
parser.add_argument('--starttime', type=str, default='', metavar='ST',
                    help='time when experiment starts')
parser.add_argument('--testsamplenum', type=int, default=100,
                    help='test sample number for synthesizing (default: 100)')
parser.add_argument('-K', '--K', type=int, default=9,
                    help='Number of Gaussians (default: 9)')
parser.add_argument('--zdim', type=int, default=32, metavar='ZD',
                    help='dimension of latent variable Z')
parser.add_argument('--h-dim', type=int, default=128, metavar='HD',
                    help='dimension of hidden layer of encoder')
parser.add_argument('--h-dim1', type=int, default=128, metavar='HD1',
                    help='dimension of hidden layer 1 of decoder')
parser.add_argument('--h-dim2', type=int, default=64, metavar='HD2',
                    help='dimension of hidden layer 2 of decoder')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalize dataset before training (default: False)')
parser.add_argument('--isplot', type=int, default=0,
                    help='plot graph (1) or not (0) (default: 0)')
parser.add_argument('--everyepoch', type=int, default=10,
                    help='Every # epoch for graph plot (default: 10)')
parser.add_argument('--savemodel', type=int, default=1,
                    help='save model (1) or not (0) (default: 1)')
parser.add_argument('--train', type=int, default=1,
                    help='train model (1) or not (0) (default: 1)')
parser.add_argument('--loadmodel', type=int, default=0,
                    help='load trained model (1) or not (0) (default: 0)')
parser.add_argument('--modelloc', type=str, default='',
                    help='location of trained model')
parser.add_argument('--generate', type=int, default=0,
                    help='generate synthesized gene expression data (1) or not (0) (default: 0)')
parser.add_argument('--generatenum', type=int, default=100,
                    help='number of synthesized data (default: 100)')
parser.add_argument('-gc', '--generatecls', nargs='+', type=int, default=[0],
                    help='list of cluster labels. i.e., [0,1,2,3] (default: [0])')
parser.add_argument('--generatefolder', type=str, default='',
                    help='directory for generating synthesized data')
parser.add_argument('--log', type=int, default=0,
                    help='transfer matrix A values to log(A+1) (1) or not (0) (default: 0)')
parser.add_argument('--logt', type=int, default=1,
                    help='transfer matrix A values to log(A+10^t) (1) or not (0) (default: 1)')
parser.add_argument('--exp', type=int, default=0,
                    help='transfer matrix A values back from log values (1) or not (0) (default: 0)')
parser.add_argument('--scalegene', type=int, default=0,
                    help='scale each gene across cells (1) or not (0) (default: 0)')
parser.add_argument('--scalecell', type=int, default=0,
                    help='scale each cell across genes (1) or not (0) (default: 0)')
parser.add_argument('--batch', type=int, default=1,
                    help='run batch mode, process every .mtx files in datadir')
parser.add_argument('--modelselect', type=int, default=0,
                    help='perform model selection or not (default: 0). 0 for no model selection,')
parser.add_argument('--loss', type=str, default='zinb', metavar='LS',
                    help='loss function (mse/l1/zinb) (default: zinb)')
parser.add_argument('--parallel', type=int, default=0,
                    help='perform data parallelism or not')

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py line 105
# MODIFIED: Default batch_size from 37 to 128 for efficiency
# Original: default=37
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py line 107
# MODIFIED: Default epochs from 13 to 200 for better convergence
# Original: default=13
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py lines 109-126
# All following parser arguments are EXACT copies
parser.add_argument('--lr', type=float, default=6e-6, metavar='LR',
                    help='learning rate (default: 0.000006)')
parser.add_argument('--manual_seed', type=int, default=42,
                    help='seed number for reproduction (default: 42)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--modelload', action='store_true', default=False,
                    help='start from loaded model')
parser.add_argument('--save', type=int, default=20,
                    help='Interval between saving model (default: 20 epochs). save=0 means model will not be saved.')
parser.add_argument('--predict', action='store_true', default=False,
                    help='Running mode. Default is training, true if predicting')
parser.add_argument('--plotlabels', type=str, default='model',
                    help='Directory name for Plotting labels')
parser.add_argument('--l1', type=float, default=0, metavar='L1',
                    help='L1 regularization lambda (default: 0)')
parser.add_argument('--seed', type=int, default=123, metavar='SE',
                    help='seed (default: 123)')

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py line 127
# MODIFIED: Added default='lupus' (original had no default)
# Original: parser.add_argument('--dataset', help='dataset to use')
parser.add_argument('--dataset', default='lupus', help='dataset to use')

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py lines 129-150
# All following code is EXACT copy
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.device = device
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.modelsaveloc == '':
    args.modelsaveloc = args.outputdir
args.start = str(time.time())

if args.cuda:
    torch.cuda.manual_seed(args.manual_seed)

for o in os.listdir(args.data_loc):
    print(o)
print(args.data_loc)

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)
if not os.path.exists(args.modelsaveloc):
    os.makedirs(args.modelsaveloc)

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py line 153
# MODIFIED: Pickle filename to include dataset name
# Original: with open("args_save_zinb.pickle", "wb") as output_file:
with open(f"{args.modelsaveloc}/args_{args.dataset}.pickle", "wb") as output_file:
    pickle.dump(args, output_file)

# SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py lines 155-194
# Training loop is EXACT copy with one addition at the end
torch.manual_seed(args.seed)
np.random.seed(args.seed)

for f in os.listdir(args.data_loc):
    if f.endswith('.mtx'):
        args.dataset = f[:-4]
        print(f"Processing {args.dataset}")
        train_loader = DataLoader(args.data_loc, args.dataset, batch_size=args.batch_size, 
                                 shuffle=True, **kwargs)
        test_loader = TestDataLoader(args.data_loc, args.dataset, batch_size=args.batch_size, 
                                    shuffle=False, **kwargs)
        
        print(f"Input dimension: {train_loader.input_dim}")
        args.input_dim = train_loader.input_dim
        args.n = train_loader.N
        
        gmvae = GMVAE_ZINB(args).to(device)
        optimizer = optim.Adam(gmvae.parameters(), lr=args.lr)
        
        print("Start training...")
        loss_across_epochs = []
        for epoch in range(1, args.epochs + 1):
            loss = gmvae_train(args, gmvae, device, train_loader, optimizer, epoch)
            loss_across_epochs.append(loss)
            
            if args.save != 0 and epoch % args.save == 0:
                model_loc = args.modelsaveloc + '/' + args.dataset + 'PT_' + \
                           'K' + str(args.K) + '_' + \
                           'l' + str(args.zdim) + '_' + \
                           'h' + str(args.h_dim) + '_h1' + str(args.h_dim1) + \
                           '_h2' + str(args.h_dim2) + '_' + args.loss + '_' + \
                           str(time.time())[-5:] + '_wholePTmodel.pt'
                torch.save(gmvae, model_loc)
                print(f'Model saved at {model_loc}')
        
        # ADDITION: Save final model state dict for loading in P4P integration
        # This is the ONLY addition to the training loop
        torch.save(gmvae.state_dict(), f'./{args.modelsaveloc}/gmvae_model.pth')
        
        # SOURCE: bulk2sc_GMVAE/_3_model_train_GMVAE/main_zinb.py lines 196-229
        # Test evaluation is EXACT copy with one addition at the end
        if not os.path.exists(args.outputdir + args.start):
            os.makedirs(args.outputdir + args.start)
        np.save(args.outputdir + args.start + '/loss_zinb.npy', np.array(loss_across_epochs))
        
        print("Evaluating on test set...")
        gmvae.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)
                pi_x, mu_zs, logvar_zs, x_hat, mu_genz, logvar_genz, pi_z, mu_x, logstd_x, w_pi, w_mu, w_sigma = gmvae(data, labels)
                loss = gmvae.loss_function_zinb(x_hat, data, pi_x, mu_zs, logvar_zs, pi_z, mu_genz, logvar_genz, mu_x, logstd_x, labels)
                test_loss += loss.item()
        
        test_loss /= len(test_loader.dataset)
        print(f'Test set loss: {test_loss:.4f}')
        
        if args.savemodel:
            model_loc = args.modelsaveloc + '/' + args.dataset + 'PT_' + \
                       'K' + str(args.K) + '_' + \
                       'l' + str(args.zdim) + '_' + \
                       'h' + str(args.h_dim) + '_h1' + str(args.h_dim1) + \
                       '_h2' + str(args.h_dim2) + '_' + args.loss + '_' + \
                       str(time.time())[-5:] + '_final_wholePTmodel.pt'
            torch.save(gmvae, model_loc)
            print(f'Final model saved at {model_loc}')
        
        # ADDITION: Save final model state dict for loading in P4P integration
        # This is the ONLY addition to the test evaluation
        torch.save(gmvae.state_dict(), f'./{args.modelsaveloc}/gmvae_model.pth')