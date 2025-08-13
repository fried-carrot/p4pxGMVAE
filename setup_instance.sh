#!/bin/bash
# GCP Instance Setup Script for GMVAE-P4P Training

set -e  # Exit on error

echo "==================================="
echo "GMVAE-P4P GCP Instance Setup"
echo "==================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3-pip \
    git \
    htop \
    tmux \
    build-essential \
    python3.9-dev

# Create project directory
echo "Setting up project directory..."
cd ~
mkdir -p gmvae-p4p-project
cd gmvae-p4p-project

# Create virtual environment
echo "Creating Python virtual environment..."
python3.9 -m venv gmvae_env
source gmvae_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Detect if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. Installing PyTorch with CUDA support..."
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
else
    echo "No GPU detected. Installing CPU-only PyTorch..."
    pip install torch==2.0.1 torchvision==0.15.2
fi

# Install required packages
echo "Installing required Python packages..."
pip install \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    scanpy==1.9.3 \
    anndata==0.9.1 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    tqdm==4.65.0 \
    h5py==3.9.0

# Clone repository (replace with your repo)
echo "Cloning repository..."
if [ -d "p4pxGMVAE" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd p4pxGMVAE
    git pull
else
    # Replace with your actual repository URL
    git clone https://github.com/YOUR_USERNAME/p4pxGMVAE.git
    cd p4pxGMVAE
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p data/lupus/h5ad
mkdir -p data_processed/lupus_gmvae
mkdir -p models
mkdir -p results
mkdir -p logs

# Set up environment variables
echo "Setting up environment variables..."
cat > ~/.gmvae_env << 'EOF'
export GMVAE_HOME=~/gmvae-p4p-project/p4pxGMVAE
export PYTHONPATH=$GMVAE_HOME:$PYTHONPATH
alias activate_gmvae='source ~/gmvae-p4p-project/gmvae_env/bin/activate'
EOF

echo "source ~/.gmvae_env" >> ~/.bashrc

# Create helper scripts
echo "Creating helper scripts..."

# Create monitoring script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
if command -v nvidia-smi &> /dev/null; then
    watch -n 1 nvidia-smi
else
    echo "Monitoring CPU and memory usage..."
    htop
fi
EOF
chmod +x monitor_training.sh

# Create training launcher script
cat > run_training.sh << 'EOF'
#!/bin/bash
source ~/gmvae-p4p-project/gmvae_env/bin/activate
cd ~/gmvae-p4p-project/p4pxGMVAE

# Check if data exists
if [ ! -f "data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad" ]; then
    echo "Error: Lupus data not found!"
    echo "Please upload data to: data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad"
    exit 1
fi

# Prepare data
echo "Preparing lupus data..."
python prepare_lupus_data_vm.py \
    --input data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad \
    --output data_processed/lupus_gmvae \
    --chunk-size 50000

# Train GMVAE
echo "Training GMVAE..."
python train_gmvae_lupus_vm.py \
    --data-loc data_processed/lupus_gmvae/ \
    --epochs 200 \
    --batch-size 256 \
    --auto-batch-size \
    2>&1 | tee logs/gmvae_training.log

# Train ProtoCell with GMVAE
echo "Training ProtoCell with GMVAE..."
python protocell_gmvae_vm.py \
    --data-path data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad \
    --gmvae-model-path models/gmvae_lupus_epoch200.pth \
    --epochs 100 \
    --batch-size 8 \
    2>&1 | tee logs/protocell_training.log

echo "Training complete!"
EOF
chmod +x run_training.sh

# Create data download script
cat > download_data.sh << 'EOF'
#!/bin/bash
# Download lupus data from GCS (update bucket name)
BUCKET_NAME="gs://your-bucket-name"

echo "Downloading data from $BUCKET_NAME..."
gsutil -m cp -r $BUCKET_NAME/lupus_data/* data/lupus/h5ad/

# Alternative: Download from a public URL if available
# wget -O data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad "YOUR_DATA_URL"
EOF
chmod +x download_data.sh

# Create results upload script
cat > upload_results.sh << 'EOF'
#!/bin/bash
BUCKET_NAME="gs://your-bucket-name"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Uploading results to $BUCKET_NAME..."
gsutil -m cp -r models/* $BUCKET_NAME/results/$TIMESTAMP/models/
gsutil -m cp -r results/* $BUCKET_NAME/results/$TIMESTAMP/results/
gsutil -m cp logs/*.log $BUCKET_NAME/results/$TIMESTAMP/logs/

echo "Results uploaded to $BUCKET_NAME/results/$TIMESTAMP/"
EOF
chmod +x upload_results.sh

echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Upload your data using: ./download_data.sh"
echo "2. Start training in tmux:"
echo "   tmux new -s training"
echo "   ./run_training.sh"
echo "3. Monitor GPU/CPU in another terminal:"
echo "   ./monitor_training.sh"
echo "4. Upload results when done:"
echo "   ./upload_results.sh"
echo ""
echo "To activate the virtual environment:"
echo "   source ~/gmvae-p4p-project/gmvae_env/bin/activate"
echo ""
echo "GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No GPU detected - will use CPU"
fi