#!/bin/bash

PROJECT_ID="your-project-id"
ZONE="us-central1-a"
INSTANCE_NAME="gmvae-trainer"
BUCKET_NAME="gmvae-models"

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=n1-highmem-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --preemptible \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --metadata="install-nvidia-driver=True" \
    --metadata-from-file startup-script=gcp_startup.sh

echo "Instance created. Connect with:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"