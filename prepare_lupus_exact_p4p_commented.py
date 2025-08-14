#!/usr/bin/env python
"""
DATA PREPROCESSING FOR LUPUS DATASET - EXACT COPY FROM P4P
All preprocessing steps are EXACT copies from ProtoCell4P/src/load_data.py

SOURCE: ProtoCell4P/src/load_data.py, function load_lupus (lines 25-75)
GitHub: https://github.com/deepomicslab/ProtoCell4P
"""
import scanpy as sc
import pandas as pd
import scipy.io as sio
import numpy as np
import os
from pathlib import Path

def prepare_gmvae_data(input_h5ad, output_dir, subsample_fraction=None):
    """
    Prepare lupus data using EXACT P4P preprocessing pipeline
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # SOURCE: ProtoCell4P/src/load_data.py line 29
    # Original: adata = sc.read_h5ad(data_path)
    adata = sc.read_h5ad(input_h5ad)
    
    # ADDITION: Subsampling for faster testing (NOT in P4P)
    # This is the ONLY addition not from P4P - used for testing efficiency
    if subsample_fraction is not None:
        n_cells = adata.shape[0]
        n_subsample = int(n_cells * subsample_fraction)
        np.random.seed(42)
        indices = np.random.choice(n_cells, n_subsample, replace=False)
        adata = adata[indices, :]
        print(f"Subsampled to {subsample_fraction*100}%: {n_subsample} cells from {n_cells}")
    
    # SOURCE: ProtoCell4P/src/load_data.py line 31-32
    # Original comment: "# before: (834096, 32738) | after: (834096, 24205)"
    # Original: sc.pp.filter_genes(adata, min_cells=5)
    print(f"Before filtering: {adata.shape[1]} genes")
    sc.pp.filter_genes(adata, min_cells=5)
    print(f"After filtering (min_cells=5): {adata.shape[1]} genes")
    
    # SOURCE: ProtoCell4P/src/load_data.py line 33
    # Original: sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("Normalized to target_sum=1e4")
    
    # NOTE: ProtoCell4P/src/load_data.py line 34 is COMMENTED OUT:
    # "# sc.pp.scale(adata, max_value=10, zero_center=True) # Unable to allocate 75.2 GiB"
    # We follow P4P and DO NOT scale
    
    # SOURCE: ProtoCell4P/src/load_data.py lines 36-37
    # Original: if keep_sparse is False: adata.X = adata.X.toarray()
    # We keep sparse format for memory efficiency
    X = adata.X
    
    if not hasattr(X, 'tocsr'):
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.array(X))
    
    # Convert to genes x cells format for GMVAE (transpose)
    X_transposed = X.T.tocsr()
    sio.mmwrite(os.path.join(output_dir, "matrix.mtx"), X_transposed)
    
    # SOURCE: ProtoCell4P/src/load_data.py line 46
    # Original: cell_types = adata.obs["ct_cov"]
    # P4P specifically uses "ct_cov" column for lupus cell types
    cell_types = adata.obs["ct_cov"]
    
    # SOURCE: ProtoCell4P/src/load_data.py lines 48-49
    # Original: ct_id = sorted(set(cell_types))
    # Original: mapping_ct = {c:idx for idx, c in enumerate(ct_id)}
    cell_type_codes = pd.Categorical(cell_types).codes
    pd.DataFrame({'cluster': cell_type_codes}).to_csv(
        os.path.join(output_dir, "labels.csv"), index=False
    )
    
    # SOURCE: ProtoCell4P/src/load_data.py line 44
    # Original: genes = adata.var_names.tolist()
    with open(os.path.join(output_dir, "genes.txt"), 'w') as f:
        for gene in adata.var_names:
            f.write(f"{gene}\n")
    
    print(f"Prepared data: {X_transposed.shape[0]} genes x {X_transposed.shape[1]} cells")
    print(f"Cell types: {len(np.unique(cell_type_codes))}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Default path matches P4P's expected location
    # SOURCE: ProtoCell4P/src/load_data.py line 25
    # Original default: "../data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad"
    parser.add_argument('--input', type=str, 
                       default='h5ad/CLUESImmVar_nonorm.V6.h5ad')
    parser.add_argument('--output', type=str, 
                       default='data_processed/lupus_gmvae')
    
    # ADDITION: Subsampling option for testing (NOT in P4P)
    parser.add_argument('--subsample', type=float, default=None, 
                       help='Fraction to subsample for faster testing')
    args = parser.parse_args()
    
    prepare_gmvae_data(args.input, args.output, subsample_fraction=args.subsample)