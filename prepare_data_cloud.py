#!/usr/bin/env python

import os
import argparse
import scanpy as sc
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='data_processed/lupus_gmvae')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading {args.input}...")
    adata = sc.read_h5ad(args.input)
    
    print(f"Original shape: {adata.shape}")
    
    if adata.raw is not None:
        adata = adata.raw.to_adata()
    
    if 'ct_cov' in adata.obs.columns:
        cell_types = adata.obs['ct_cov']
    elif 'cell_type' in adata.obs.columns:
        cell_types = adata.obs['cell_type']
    elif 'celltype' in adata.obs.columns:
        cell_types = adata.obs['celltype']
    else:
        raise ValueError("No cell type column found")
    
    unique_types = sorted(cell_types.unique())
    type_to_id = {ct: i for i, ct in enumerate(unique_types)}
    
    print(f"Found {len(unique_types)} cell types:")
    for i, ct in enumerate(unique_types):
        count = (cell_types == ct).sum()
        print(f"  {i}: {ct} ({count} cells)")
    
    cluster_ids = [type_to_id[ct] for ct in cell_types]
    
    X = adata.X
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)
    
    X_transposed = X.T
    
    print(f"Saving matrix ({X_transposed.shape[0]} genes × {X_transposed.shape[1]} cells)...")
    mmwrite(os.path.join(args.output, 'matrix.mtx'), X_transposed)
    
    labels_df = pd.DataFrame({'cluster': cluster_ids})
    labels_df.to_csv(os.path.join(args.output, 'labels.csv'), index=False)
    
    metadata = {
        'n_genes': X_transposed.shape[0],
        'n_cells': X_transposed.shape[1],
        'n_clusters': len(unique_types),
        'cell_types': unique_types
    }
    pd.Series(metadata).to_json(os.path.join(args.output, 'metadata.json'))
    
    print("Done")

if __name__ == '__main__':
    main()