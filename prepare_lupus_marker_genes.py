#!/usr/bin/env python
import scanpy as sc
import pandas as pd
import scipy.io as sio
import numpy as np
import os
from pathlib import Path

# P4P marker genes for lupus dataset (from paper)
LUPUS_MARKER_GENES = [
    'CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B',  # T cells
    'CD19', 'MS4A1', 'CD79A', 'CD79B',  # B cells  
    'NCAM1', 'NKG7', 'GNLY', 'KLRD1',  # NK cells
    'CD14', 'LYZ', 'S100A8', 'S100A9',  # Monocytes
    'FCGR3A', 'MS4A7', 'CDKN1C',  # FCGR3A+ Monocytes
    'FCER1A', 'CST3', 'IL3RA',  # Dendritic cells
    'PPBP', 'PF4', 'GP9',  # Megakaryocytes
    'IL7R', 'CCR7', 'LEF1', 'TCF7',  # Memory T cells
    'HLA-DRA', 'HLA-DRB1', 'HLA-DQA1'  # MHC II
]

def prepare_gmvae_data(input_h5ad, output_dir, use_raw=True, subsample_fraction=None, use_marker_genes=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    adata = sc.read_h5ad(input_h5ad)
    
    # Subsample if requested
    if subsample_fraction is not None:
        n_cells = adata.shape[0]
        n_subsample = int(n_cells * subsample_fraction)
        np.random.seed(42)
        indices = np.random.choice(n_cells, n_subsample, replace=False)
        adata = adata[indices, :]
        print(f"Subsampled to {subsample_fraction*100}%: {n_subsample} cells from {n_cells}")
    
    # Use only marker genes if requested
    if use_marker_genes:
        available_markers = [g for g in LUPUS_MARKER_GENES if g in adata.var_names]
        print(f"Using {len(available_markers)} marker genes out of {len(LUPUS_MARKER_GENES)} requested")
        print(f"Missing genes: {set(LUPUS_MARKER_GENES) - set(available_markers)}")
        adata = adata[:, available_markers]
        print(f"Reduced from {len(adata.var_names)} to {len(available_markers)} genes")
    
    X = adata.raw.X if use_raw and adata.raw and not use_marker_genes else adata.X
    
    if not hasattr(X, 'tocsr'):
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.array(X))
    
    X_transposed = X.T.tocsr()
    sio.mmwrite(os.path.join(output_dir, "matrix.mtx"), X_transposed)
    
    if 'cell_type' in adata.obs.columns:
        cell_types = adata.obs['cell_type']
    elif 'celltype' in adata.obs.columns:
        cell_types = adata.obs['celltype']
    elif 'ct_cov' in adata.obs.columns:
        cell_types = adata.obs['ct_cov']
    else:
        raise ValueError("No cell type column found")
    
    cell_type_codes = pd.Categorical(cell_types).codes
    pd.DataFrame({'cluster': cell_type_codes}).to_csv(os.path.join(output_dir, "labels.csv"), index=False)
    
    with open(os.path.join(output_dir, "genes.txt"), 'w') as f:
        for gene in adata.var_names:
            f.write(f"{gene}\n")
    
    print(f"Prepared data: {X_transposed.shape[0]} genes x {X_transposed.shape[1]} cells")
    print(f"Cell types: {len(np.unique(cell_type_codes))}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='h5ad/CLUESImmVar_nonorm.V6.h5ad')
    parser.add_argument('--output', type=str, default='data_processed/lupus_gmvae')
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--subsample', type=float, default=None)
    parser.add_argument('--no-markers', action='store_true', help='Use all genes instead of markers')
    args = parser.parse_args()
    
    prepare_gmvae_data(args.input, args.output, use_raw=not args.normalized, 
                      subsample_fraction=args.subsample,
                      use_marker_genes=not args.no_markers)