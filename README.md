# kirjuri

GPU Powered Whisper Transcription

## TODO

- [ ] Backend core functions
    - [x] Read files from Google Cloud Storage
    - [x] Remove files from Google Cloud Storage
    - [x] Write transcription results to Google Cloud Storage
    - [x] Add diarization
    - [x] Improve output format
    - [x] Email notification on complete: Using Mailjet
    - [ ] GPT summaries
    - [ ] Basic error handling: Do not fail due to single file/segment/word transcription failure
- [ ] Frontend core functions
    - [x] Create UUID for each upload set: `<bucket>/<uuid>/<audio-files + config.json>`
    - [x] Display transcription results for given UUID
    - [ ] Estimate completion time
    - [x] Visualize per-word scores in frontend to show transcription certainty
- [ ] CI/CD pipelines GitHub Actions
    - [ ] Backend transcription engine docker build + deploy to Artifact Registry
    - [ ] Backend Workflow deploy
    - [ ] Frontend Docker build + deploy to Artifact Registry
    - [ ] Frontend Google Cloud Run deploy
- [ ] Transition to Infrastructure as Code
    - [ ] GCP IAM
    - [ ] GCP Storage
    - [ ] GCP Workflow
    - [ ] GCP Scheduler
    - [ ] GCP Cloud Run
    - [ ] GCP Cloud Run Domain Mapping
- [ ] Monitoring
    - [x] Add Cloud Logging to backend container application
    - [ ] Add Cloud Logging to backend startup script
    - [ ] Add Cloud Logging to frontend container application
- [x] Devcontainer support
    - [x] Backend
    - [x] Frontend
- [ ] Optimization
    - [ ] Cache container and model files to VM disk for faster startup time
    - [ ] Slim down backend Docker container if possible
- [ ] Auth0 integration
- [x] Allow recording audio in the frontend
- [x] Purchase domain (kirjuri.net)
- [x] Redirect Cloud Run to domain (kirjuri.net)
- [x] Fix workflow bug: Do not do anything if instance is already running - do not delete running instance by accident
