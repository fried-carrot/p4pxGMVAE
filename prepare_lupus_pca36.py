#!/usr/bin/env python
"""
Prepare lupus data with PCA reduction to 36 dimensions
Based on the P4P pretraining diagram showing input_dim=36
"""
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.io as sio
import os
from pathlib import Path
from sklearn.decomposition import PCA

def prepare_gmvae_data(input_h5ad, output_dir, n_components=36, subsample_fraction=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    adata = sc.read_h5ad(input_h5ad)
    
    if subsample_fraction is not None:
        n_cells = adata.shape[0]
        n_subsample = int(n_cells * subsample_fraction)
        np.random.seed(42)
        indices = np.random.choice(n_cells, n_subsample, replace=False)
        adata = adata[indices, :]
        print(f"Subsampled to {subsample_fraction*100}%: {n_subsample} cells from {n_cells}")
    
    # P4P preprocessing
    print(f"Before filtering: {adata.shape[1]} genes")
    sc.pp.filter_genes(adata, min_cells=5)
    print(f"After filtering (min_cells=5): {adata.shape[1]} genes")
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("Normalized to target_sum=1e4")
    
    # Convert to dense if sparse
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # PCA to 36 dimensions (as shown in P4P diagram)
    print(f"Applying PCA: {X.shape[1]} genes -> {n_components} components")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Save PCA model for later use
    import pickle
    with open(os.path.join(output_dir, "pca_model.pkl"), 'wb') as f:
        pickle.dump(pca, f)
    
    # Convert to sparse for consistency
    from scipy.sparse import csr_matrix
    X_pca_sparse = csr_matrix(X_pca)
    
    # Transpose for GMVAE format (features x cells)
    X_transposed = X_pca_sparse.T.tocsr()
    sio.mmwrite(os.path.join(output_dir, "matrix.mtx"), X_transposed)
    
    # Cell types
    cell_types = adata.obs["ct_cov"]
    cell_type_codes = pd.Categorical(cell_types).codes
    pd.DataFrame({'cluster': cell_type_codes}).to_csv(
        os.path.join(output_dir, "labels.csv"), index=False
    )
    
    # Save component names instead of gene names
    with open(os.path.join(output_dir, "genes.txt"), 'w') as f:
        for i in range(n_components):
            f.write(f"PC{i+1}\n")
    
    print(f"Prepared data: {X_transposed.shape[0]} PCs x {X_transposed.shape[1]} cells")
    print(f"Cell types: {len(np.unique(cell_type_codes))}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='h5ad/CLUESImmVar_nonorm.V6.h5ad')
    parser.add_argument('--output', type=str, default='data_processed/lupus_gmvae_pca36')
    parser.add_argument('--n-components', type=int, default=36, 
                       help='Number of PCA components (default: 36 as in P4P diagram)')
    parser.add_argument('--subsample', type=float, default=None)
    args = parser.parse_args()
    
    prepare_gmvae_data(args.input, args.output, 
                       n_components=args.n_components,
                       subsample_fraction=args.subsample)