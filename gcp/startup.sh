#!/bin/bash

echo "Begin startup payload script"

# Install NVIDIA driver
echo "Installing NVIDIA driver .."
sudo /opt/deeplearning/install-driver.sh

# Find out the instance name and instance zone
INSTANCE_NAME=$(curl -sSH "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/name")
INSTANCE_ZONE=$(curl -sSH "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $4}')
echo "Instance name: $INSTANCE_NAME, zone: $INSTANCE_ZONE"

# Query container name from metadata
CONTAINER_NAME=$(gcloud compute instances describe $INSTANCE_NAME --zone $INSTANCE_ZONE --format='value[](metadata.items.payload-container-name)')
ARTIFACT_REGISTRY_REGION="${CONTAINER_NAME%%/*}"

echo "Configuring artifact registry region: $ARTIFACT_REGISTRY_REGION .."
gcloud auth configure-docker $ARTIFACT_REGISTRY_REGION --quiet
echo "Pulling docker container: $CONTAINER_NAME .."
docker pull $CONTAINER_NAME

# Update VM instance 'application-status' metadata to 'RUNNING'
echo "Updating application-status metadata to: 'RUNNING' .."
gcloud compute instances add-metadata $INSTANCE_NAME --metadata application-status=RUNNING --zone $INSTANCE_ZONE

echo "Running docker container"
docker run --rm --gpus all --network="host" --privileged $CONTAINER_NAME
echo "Docker container run complete"

# Update VM instance 'application-status' metadata to 'DONE'
echo "Updating application-status metadata to: 'DONE' .."
gcloud compute instances add-metadata $INSTANCE_NAME --metadata application-status=DONE --zone $INSTANCE_ZONE

echo "Startup payload script done"