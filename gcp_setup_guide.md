# Google Cloud Platform Setup for GMVAE-P4P Training

## 1. GCP Instance Recommendations

### For Full Dataset Training (834,096 cells × 32,738 genes)

#### Option A: High-Performance Training (Recommended)
```bash
# Create instance with A100 GPU
gcloud compute instances create gmvae-p4p-training \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```
- **GPU**: 1x A100 (40GB VRAM)
- **RAM**: 85GB
- **vCPUs**: 12
- **Cost**: ~$2.95/hour
- **Training time**: ~8-12 hours for 200 epochs

#### Option B: Budget-Friendly Training
```bash
# Create instance with T4 GPU
gcloud compute instances create gmvae-p4p-training \
    --zone=us-central1-a \
    --machine-type=n1-highmem-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-standard \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```
- **GPU**: 1x T4 (16GB VRAM)
- **RAM**: 52GB
- **vCPUs**: 8
- **Cost**: ~$0.50/hour
- **Training time**: ~24-36 hours for 200 epochs

#### Option C: CPU-Only (Not Recommended)
```bash
# Create CPU-only instance
gcloud compute instances create gmvae-p4p-training \
    --zone=us-central1-a \
    --machine-type=n2-highmem-16 \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-standard
```
- **RAM**: 128GB
- **vCPUs**: 16
- **Cost**: ~$0.98/hour
- **Training time**: ~120+ hours for 200 epochs

## 2. Initial Setup

### Step 1: Install Google Cloud SDK locally
```bash
# macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### Step 2: Authenticate and set project
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Step 3: Create and start your instance
```bash
# Use one of the instance creation commands above
# Then SSH into it:
gcloud compute ssh gmvae-p4p-training --zone=us-central1-a
```

## 3. Instance Setup Script

Save this as `setup_instance.sh` and run after SSHing:

```bash
#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.9 and pip
sudo apt-get install -y python3.9 python3.9-venv python3-pip git

# Create virtual environment
python3.9 -m venv gmvae_env
source gmvae_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy pandas scanpy anndata scikit-learn matplotlib seaborn tqdm

# Clone your repository
git clone https://github.com/YOUR_USERNAME/p4pxGMVAE.git
cd p4pxGMVAE

# Create necessary directories
mkdir -p data_processed/lupus_gmvae
mkdir -p models
mkdir -p results
```

## 4. Data Transfer

### Option A: Using Google Cloud Storage (Recommended)
```bash
# On your local machine, upload data to GCS
gsutil mb gs://gmvae-p4p-data
gsutil -m cp -r data/* gs://gmvae-p4p-data/

# On the GCP instance, download data
gsutil -m cp -r gs://gmvae-p4p-data/* data/
```

### Option B: Direct SCP
```bash
# From local machine
gcloud compute scp --recurse data/* gmvae-p4p-training:~/p4pxGMVAE/data/ --zone=us-central1-a
```

## 5. Running Training Pipeline

### Step 1: Prepare data
```bash
source gmvae_env/bin/activate
cd ~/p4pxGMVAE

# Prepare lupus data
python prepare_lupus_data_vm.py \
    --input data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad \
    --output data_processed/lupus_gmvae \
    --chunk-size 50000
```

### Step 2: Train GMVAE
```bash
# For GPU instance
python train_gmvae_lupus_vm.py \
    --data-loc data_processed/lupus_gmvae/ \
    --epochs 200 \
    --batch-size 256 \
    --auto-batch-size

# For CPU instance (not recommended)
python train_gmvae_lupus_vm.py \
    --data-loc data_processed/lupus_gmvae/ \
    --epochs 200 \
    --batch-size 64 \
    --force-cpu \
    --memory-efficient
```

### Step 3: Train integrated ProtoCell model
```bash
python protocell_gmvae_vm.py \
    --data-path data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad \
    --gmvae-model-path models/gmvae_lupus_epoch200.pth \
    --epochs 100 \
    --batch-size 8
```

## 6. Monitoring and Management

### Use tmux for persistent sessions
```bash
# Start tmux session
tmux new -s training

# Run your training
python train_gmvae_lupus_vm.py --data-loc data_processed/lupus_gmvae/ --epochs 200

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### Monitor GPU usage
```bash
# In a separate tmux window
watch -n 1 nvidia-smi
```

### Monitor training logs
```bash
# Create a training script with logging
python train_gmvae_lupus_vm.py ... 2>&1 | tee training.log

# Watch log in real-time
tail -f training.log
```

## 7. Cost Optimization

### Use Preemptible VMs (70% cheaper)
```bash
gcloud compute instances create gmvae-p4p-training \
    --preemptible \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB
```

### Auto-shutdown script
```bash
# Add to your training script
echo "sudo shutdown -h now" | at now + 8 hours
```

### Use spot instances for long training
```python
# Add checkpoint saving to training scripts
if epoch % 10 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_epoch_{epoch}.pth')
```

## 8. Results Transfer

### Save results to GCS
```bash
# After training completes
gsutil -m cp -r models/* gs://gmvae-p4p-data/models/
gsutil -m cp -r results/* gs://gmvae-p4p-data/results/
gsutil cp training.log gs://gmvae-p4p-data/logs/
```

### Download to local
```bash
# On local machine
gsutil -m cp -r gs://gmvae-p4p-data/models/* ./models/
gsutil -m cp -r gs://gmvae-p4p-data/results/* ./results/
```

## 9. Clean Up

### Stop instance (keeps disk)
```bash
gcloud compute instances stop gmvae-p4p-training --zone=us-central1-a
```

### Delete instance (removes everything)
```bash
gcloud compute instances delete gmvae-p4p-training --zone=us-central1-a
```

## 10. Estimated Costs

| Configuration | Hourly Cost | 200 Epochs | 500 Epochs |
|--------------|-------------|------------|------------|
| A100 GPU | $2.95 | ~$35 (12h) | ~$88 (30h) |
| T4 GPU | $0.50 | ~$18 (36h) | ~$45 (90h) |
| T4 Preemptible | $0.15 | ~$5.40 (36h) | ~$13.50 (90h) |
| CPU-only | $0.98 | ~$118 (120h) | ~$294 (300h) |

## Quick Start Commands

```bash
# 1. Create instance (T4 GPU, preemptible for budget)
gcloud compute instances create gmvae-training \
    --zone=us-central1-a \
    --machine-type=n1-highmem-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --preemptible \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --metadata="install-nvidia-driver=True"

# 2. SSH and setup
gcloud compute ssh gmvae-training --zone=us-central1-a

# 3. Run setup script (create and run setup_instance.sh from above)

# 4. Start training in tmux
tmux new -s train
python prepare_lupus_data_vm.py --input data/lupus.h5ad --output data_processed/
python train_gmvae_lupus_vm.py --data-loc data_processed/lupus_gmvae/ --epochs 200

# 5. Monitor and manage
# Detach: Ctrl+B, D
# Check later: gcloud compute ssh gmvae-training --zone=us-central1-a
# Reattach: tmux attach -t train
```