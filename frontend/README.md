# Kirjuri frontend

## Docker

### Build

```
docker build --platform linux/amd64 -t europe-west4-docker.pkg.dev/kirjuri/kirjuri-frontend/kirjuri-frontend .
```

### Run

```
docker run -p 8080:8080 europe-west4-docker.pkg.dev/kirjuri/kirjuri-frontend/kirjuri-frontend
```

```
docker run -p 8080:8080 -v $(pwd)/src:/app -v $(pwd)/.secrets/service-account.json:/app/.secrets/service-account.json -e GOOGLE_APPLICATION_CREDENTIALS=/app/.secrets/service-account.json -e BASIC_AUTH_USERNAME=<username> -e BASIC_AUTH_PASSWORD=<password> europe-west4-docker.pkg.dev/kirjuri/kirjuri-frontend/kirjuri-frontend
```

### Deploy

```
docker push europe-west4-docker.pkg.dev/kirjuri/kirjuri-frontend/kirjuri-frontend:latest
```

## Google Cloud Run

```
gcloud run deploy kirjuri-frontend \
  --image=europe-west4-docker.pkg.dev/kirjuri/kirjuri-frontend/kirjuri-frontend \
  --region=europe-west4 \
  --set-env-vars="BASIC_AUTH_USERNAME=<username>,BASIC_AUTH_PASSWORD=<password>"
```