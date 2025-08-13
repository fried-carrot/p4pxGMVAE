# GMVAE-P4P Training Pipeline

## Setup for RTX 3060 (15% subsample, 250 epochs)

### Requirements
- RTX 3060 (12GB VRAM)
- 16GB+ RAM
- CUDA 11.8

### Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas scanpy anndata tqdm scikit-learn
```

### Running
```bash
# Full pipeline
bash run_3060.sh

# Or step by step:
python prepare_data_3060.py --input data/lupus.h5ad
python train_gmvae_3060.py --epochs 250 --batch-size 64
python train_protocell_3060.py --epochs 100
```

### Files
- `prepare_data_3060.py`: Subsamples to 15% and converts to MTX
- `train_gmvae_3060.py`: Trains GMVAE for 250 epochs
- `train_protocell_3060.py`: Trains ProtoCell classifier
- `run_3060.sh`: Runs complete pipeline

Estimated time: 12-15 hours on RTX 3060