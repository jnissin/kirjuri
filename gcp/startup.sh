#!/bin/bash

CONTAINER_URL="<container-url>"

# Update TPU VM instance 'application-status' metadata to 'RUNNING'
curl -X PUT "http://metadata.google.internal/computeMetadata/v1/instance/attributes/application-status" \
  -H "Metadata-Flavor: Google" \
  -d "RUNNING"

docker pull $CONTAINER_URL
docker run --rm --network="host" --privileged $CONTAINER_URL

# Update TPU VM instance 'application-status' metadata to 'DONE'
curl -X PUT "http://metadata.google.internal/computeMetadata/v1/instance/attributes/application-status" \
  -H "Metadata-Flavor: Google" \
  -d "DONE"
