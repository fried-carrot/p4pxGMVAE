#!/bin/bash

cd /home
sudo apt-get update
sudo apt-get install -y python3-pip git

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy scipy pandas scanpy anndata scikit-learn tqdm

git clone https://github.com/YOUR_REPO/p4pxGMVAE.git
cd p4pxGMVAE

mkdir -p data_processed/lupus_gmvae
mkdir -p models

gsutil cp gs://YOUR_BUCKET/lupus_data.h5ad data/

python3 prepare_data_cloud.py --input data/lupus_data.h5ad --output data_processed/lupus_gmvae

python3 train_gmvae_cloud.py \
    --data-loc data_processed/lupus_gmvae/ \
    --epochs 200 \
    --batch-size 256 \
    --output-dir models/

gsutil cp models/gmvae_lupus_final.pth gs://YOUR_BUCKET/models/