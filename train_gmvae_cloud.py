#!/usr/bin/env python

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GMVAE_model_zinb import GMVAE
from GMVAE_losses_zinb import gmvae_train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-loc', type=str, default='data_processed/lupus_gmvae/', 
                        help='Location of data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-topic', type=int, default=8)
    parser.add_argument('--encoder1-units', type=int, default=256)
    parser.add_argument('--encoder2-units', type=int, default=128) 
    parser.add_argument('--zdim', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--print-each', type=int, default=10)
    parser.add_argument('--save-each', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='models/')
    return parser.parse_args()

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, idx):
        cell_data = self.data[:, idx].toarray().flatten()
        label = self.labels[idx] if self.labels is not None else 0
        return torch.FloatTensor(cell_data), torch.LongTensor([label])

def load_data(data_loc):
    matrix_file = os.path.join(data_loc, 'matrix.mtx')
    labels_file = os.path.join(data_loc, 'labels.csv')
    
    data = mmread(matrix_file)
    data = csr_matrix(data)
    print(f"Data shape: {data.shape}")
    
    labels_df = pd.read_csv(labels_file)
    labels = labels_df['cluster'].values
    print(f"Loaded {len(labels)} cells with {len(np.unique(labels))} cell types")
    
    return data, labels

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    X_train, y_train = load_data(args.data_loc)
    
    n_genes = X_train.shape[0]
    n_cells = X_train.shape[1]
    
    print(f"Dataset: {n_genes} genes, {n_cells} cells")
    
    train_dataset = SparseDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=0)
    
    gmvae = GMVAE(
        n_gene=n_genes,
        n_topic=args.n_topic,
        n_sample=n_cells,
        device=device
    ).to(device)
    
    optimizer = optim.Adam(gmvae.parameters(), lr=args.learning_rate)
    
    print(f"Training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        total_loss_epoch, KLD_gaussian_epoch, KLD_pi_epoch, zinb_loss_epoch, accuracy = \
            gmvae_train(epoch, gmvae, train_loader, optimizer)
        
        if epoch % args.print_each == 0:
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Loss: {total_loss_epoch:.4f} | "
                  f"ZINB: {zinb_loss_epoch:.4f} | "
                  f"KLD_g: {KLD_gaussian_epoch:.4f} | "
                  f"KLD_pi: {KLD_pi_epoch:.4f}")
        
        if epoch % args.save_each == 0:
            checkpoint_path = os.path.join(args.output_dir, f'gmvae_lupus_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': gmvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss_epoch,
            }, checkpoint_path)
    
    final_path = os.path.join(args.output_dir, 'gmvae_lupus_final.pth')
    torch.save(gmvae.state_dict(), final_path)
    print(f"Saved: {final_path}")

if __name__ == '__main__':
    main()