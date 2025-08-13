#!/bin/bash
# Quick Start Script for GCP GMVAE-P4P Training

# Configuration
PROJECT_ID="your-project-id"  # CHANGE THIS
ZONE="us-central1-a"
INSTANCE_NAME="gmvae-p4p-trainer"
BUCKET_NAME="gmvae-p4p-data"  # CHANGE THIS

echo "==================================="
echo "GCP Quick Start for GMVAE-P4P"
echo "==================================="

# Function to create instance
create_instance() {
    echo "Select instance type:"
    echo "1) A100 GPU (Best performance, ~$2.95/hr)"
    echo "2) T4 GPU (Budget-friendly, ~$0.50/hr)"
    echo "3) T4 GPU Preemptible (Most economical, ~$0.15/hr)"
    echo "4) CPU Only (Not recommended, ~$0.98/hr)"
    read -p "Enter choice [1-4]: " choice

    case $choice in
        1)
            echo "Creating A100 instance..."
            gcloud compute instances create $INSTANCE_NAME \
                --zone=$ZONE \
                --machine-type=a2-highgpu-1g \
                --accelerator=type=nvidia-tesla-a100,count=1 \
                --image-family=pytorch-latest-gpu \
                --image-project=deeplearning-platform-release \
                --boot-disk-size=200GB \
                --boot-disk-type=pd-ssd \
                --maintenance-policy=TERMINATE \
                --metadata="install-nvidia-driver=True"
            ;;
        2)
            echo "Creating T4 instance..."
            gcloud compute instances create $INSTANCE_NAME \
                --zone=$ZONE \
                --machine-type=n1-highmem-8 \
                --accelerator=type=nvidia-tesla-t4,count=1 \
                --image-family=pytorch-latest-gpu \
                --image-project=deeplearning-platform-release \
                --boot-disk-size=200GB \
                --boot-disk-type=pd-standard \
                --maintenance-policy=TERMINATE \
                --metadata="install-nvidia-driver=True"
            ;;
        3)
            echo "Creating T4 Preemptible instance..."
            gcloud compute instances create $INSTANCE_NAME \
                --zone=$ZONE \
                --machine-type=n1-highmem-8 \
                --accelerator=type=nvidia-tesla-t4,count=1 \
                --preemptible \
                --image-family=pytorch-latest-gpu \
                --image-project=deeplearning-platform-release \
                --boot-disk-size=200GB \
                --boot-disk-type=pd-standard \
                --maintenance-policy=TERMINATE \
                --metadata="install-nvidia-driver=True"
            ;;
        4)
            echo "Creating CPU-only instance..."
            gcloud compute instances create $INSTANCE_NAME \
                --zone=$ZONE \
                --machine-type=n2-highmem-16 \
                --image-family=debian-11 \
                --image-project=debian-cloud \
                --boot-disk-size=200GB \
                --boot-disk-type=pd-standard
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo "Setting GCP project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Create storage bucket if it doesn't exist
echo "Creating storage bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $ZONE gs://$BUCKET_NAME/ 2>/dev/null || echo "Bucket already exists or creation failed"

# Upload local files to bucket
echo "Do you want to upload local files to GCS? (y/n)"
read -p "Choice: " upload_choice
if [ "$upload_choice" = "y" ]; then
    echo "Uploading files to GCS..."
    gsutil -m cp -r *.py gs://$BUCKET_NAME/code/
    gsutil cp setup_instance.sh gs://$BUCKET_NAME/setup/
    
    if [ -d "data" ]; then
        echo "Uploading data directory..."
        gsutil -m cp -r data/* gs://$BUCKET_NAME/data/
    fi
fi

# Create instance
echo "Creating GCP instance..."
create_instance

# Wait for instance to be ready
echo "Waiting for instance to be ready..."
sleep 30

# Copy setup files to instance
echo "Copying setup files to instance..."
gcloud compute scp setup_instance.sh $INSTANCE_NAME:~/ --zone=$ZONE
gcloud compute scp *_vm.py $INSTANCE_NAME:~/ --zone=$ZONE 2>/dev/null || true

# SSH into instance and run setup
echo "Connecting to instance and running setup..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="chmod +x setup_instance.sh && ./setup_instance.sh"

echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "To connect to your instance:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To start training:"
echo "  1. SSH into instance"
echo "  2. tmux new -s training"
echo "  3. ./run_training.sh"
echo ""
echo "To stop instance (keeps data):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete instance (removes everything):"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Bucket name: gs://$BUCKET_NAME"