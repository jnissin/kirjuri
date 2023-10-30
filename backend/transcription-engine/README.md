# Kirjuri backend: transcription-engine

The core of the kirjuri application. This includes a containerized GPU accelerated application that uses WhisperX to transcribe audio files from a specified Google Cloud Storage (GCS) bucket (ingestion). After the transcription process is complete the results are saved to another GCS bucket (results).

In order to save resources (money) the system uses a batch processing approach where the most expensive resource, Google Compute Engine instance with the GPU, is only provisioned if there are files to be processed in the ingestion bucket and the compute instance is deleted at the end of the transcription process. This process is managed via Google Cloud Scheduler and Google Cloud Workflow.

## Docker

### Build 

```
docker build --platform linux/amd64 -t europe-west4-docker.pkg.dev/kirjuri/kirjuri-transcription-engine/kirjuri-transcription-engine .
```

### Run

```
docker run --rm --gpus all europe-west4-docker.pkg.dev/kirjuri/kirjuri-transcription-engine/kirjuri-transcription-engine
```

### Deploy

```
docker push europe-west4-docker.pkg.dev/kirjuri/kirjuri-transcription-engine/kirjuri-transcription-engine:latest
```

## Workflow

### Deploy

```
gcloud workflows deploy execute-transcription-batch-run --source=gcp/workflow.yml --location europe-west4 --project kirjuri
```

### Execute

```
gcloud workflows execute execute-transcription-batch-run --location=europe-west4 --data='{"zone": "europe-west4-a", "region": "europe-west4", "instance_name": "my-gpu-instance"}'
```

## Benchmarks

The system can process (transcribe + diarize) roughly 30 seconds of audio per second using a `n1-standard-8` instance with a `NVIDIA T4` GPU. This means that **diarizing one hour of audio takes roughly 2 minutes**.