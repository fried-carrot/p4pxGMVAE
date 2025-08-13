#!/usr/bin/env python
# convert h5ad to gmvae format

import scanpy as sc
import pandas as pd
import scipy.io as sio
import numpy as np
import os
from pathlib import Path

def prepare_gmvae_data(input_h5ad, output_dir, use_raw=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    adata = sc.read_h5ad(input_h5ad)
    
    X = adata.raw.X if use_raw and adata.raw else adata.X
    
    if not hasattr(X, 'tocsr'):
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.array(X))
    
    X_transposed = X.T.tocsr()  # genes × cells
    sio.mmwrite(os.path.join(output_dir, "matrix.mtx"), X_transposed)
    
    # cell types
    if 'cell_type' in adata.obs.columns:
        cell_types = adata.obs['cell_type']
    elif 'celltype' in adata.obs.columns:
        cell_types = adata.obs['celltype']
    elif 'ct_cov' in adata.obs.columns:
        cell_types = adata.obs['ct_cov']
    else:
        raise ValueError("No cell type column found. Expected 'cell_type', 'celltype', or 'ct_cov' in adata.obs")
    
    cell_type_codes = pd.Categorical(cell_types).codes
    pd.DataFrame({'cluster': cell_type_codes}).to_csv(os.path.join(output_dir, "labels.csv"), index=False)
    
    # genes
    with open(os.path.join(output_dir, "genes.txt"), 'w') as f:
        for gene in adata.var_names:
            f.write(f"{gene}\n")
    
    # metadata
    metadata = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'n_cell_types': len(np.unique(cell_type_codes)),
        'cell_type_mapping': dict(enumerate(pd.Categorical(cell_types).categories))
    }
    
    import json
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                       default='/Users/sharatsakamuri/Documents/p4pxGMVAE/h5ad/CLUESImmVar_nonorm.V6.h5ad')
    parser.add_argument('--output', type=str, 
                       default='data_processed/lupus_gmvae')
    parser.add_argument('--normalized', action='store_true')
    args = parser.parse_args()
    
    prepare_gmvae_data(args.input, args.output, use_raw=not args.normalized)