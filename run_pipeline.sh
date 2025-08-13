#!/bin/bash

# prepare data
echo "preparing data..."
python prepare_lupus_data.py \
    --input h5ad/CLUESImmVar_nonorm.V6.h5ad \
    --output data_processed/lupus_gmvae

# train gmvae
echo "training gmvae..."
python train_gmvae_p4p.py \
    --train_gmvae \
    --data_dir data_processed/lupus_gmvae \
    --output_dir models/ \
    --gmvae_epochs 4000 \
    --gmvae_batch 128 \
    --gmvae_lr 0.001 \
    --subsample 0.1 \
    --z_dim 32

# train classifier
echo "training classifier..."
python train_gmvae_p4p.py \
    --train_classifier \
    --data_dir data_processed/lupus_gmvae \
    --gmvae_path models/gmvae_model.pth \
    --classifier_epochs 200 \
    --classifier_lr 0.001 \
    --n_classes 2 \
    --z_dim 32

echo "done"