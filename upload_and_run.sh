#!/bin/bash

# Ask for required inputs
read -p "Enter the name of the Batch Job: " JOB_NAME

PROJECT_ID="modular-cell-435907-q6"
BUCKET_NAME="sorage_lly_model"
REGION="europe-central2"
# Set Google Cloud project
echo "Setting Google Cloud project..."
gcloud config set project "$PROJECT_ID"

# Create ZIP file of the repository
echo "Compressing repository into repo.zip..."
zip -r repo.zip .

# Check if the bucket exists
if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    echo "Bucket gs://$BUCKET_NAME already exists."
else
    echo "Creating a new bucket: gs://$BUCKET_NAME..."
    gcloud storage buckets create "gs://$BUCKET_NAME" --location="$REGION"
fi

# Upload the ZIP file to Google Cloud Storage
echo "Uploading repo.zip to gs://$BUCKET_NAME..."
gsutil cp repo.zip "gs://$BUCKET_NAME/"

# Create the Batch YAML configuration
echo "Creating batch-job.yaml configuration..."
cat <<EOF > batch-job.yaml
taskGroups:
- taskSpec:
    runnable:
    - script:
        text: |
          # Install Python and dependencies
          apt-get update && apt-get install -y python3 python3-pip unzip
          
          # Download and extract the repository
          gsutil cp gs://$BUCKET_NAME/repo.zip .
          unzip repo.zip -d repo
          
          # Navigate to the example/dml directory and run the script
          cd repo/example/dml
          pip3 install -r ../../requirements.txt
          python3 main.py
    computeResource:
      cpuMilli: 2000      # 2 CPUs
      memoryMib: 4096     # 4 GB RAM
    maxRetryCount: 1
  taskCount: 1
  parallelism: 1
allocationPolicy:
  instances:
  - policy:
      machineType: e2-standard-2
logsPolicy:
  destination: CLOUD_LOGGING
EOF

# Submit the Batch Job
echo "Submitting the Batch Job: $JOB_NAME..."
gcloud batch jobs submit "$JOB_NAME" \
  --location="$REGION" \
  --config=batch-job.yaml

echo "Batch Job submitted successfully!"
echo "You can monitor the job using the following command:"
echo "  gcloud batch jobs describe $JOB_NAME --region=$REGION"
