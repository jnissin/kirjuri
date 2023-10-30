import os
import json
import uuid
import tempfile

from typing import List
from flask import Flask, render_template, request, redirect, send_from_directory
from flask_httpauth import HTTPBasicAuth
from google.cloud import storage


app = Flask(__name__)
auth = HTTPBasicAuth()

# Read the username and password from environment variables
_BASIC_AUTH_USERNAME = os.environ["BASIC_AUTH_USERNAME"]
_BASIC_AUTH_PASSWORD = os.environ["BASIC_AUTH_PASSWORD"]
_DEBUG = os.environ.get("KIRJURI_DEBUG") is not None
_USE_RELOADER = os.environ.get("KIRJURI_USE_RELOADER") is not None

users = {
    _BASIC_AUTH_USERNAME: _BASIC_AUTH_PASSWORD
}

# Initialize a Cloud Storage client and bucket access
storage_client = storage.Client()
ingestion_bucket_name = "kirjuri-ingestion-db"
ingestion_bucket = storage_client.get_bucket(ingestion_bucket_name)
results_bucket_name = "kirjuri-results-db"
results_bucket = storage_client.get_bucket(results_bucket_name)

# Create a temporary directory
transcription_results_temp_dir = tempfile.mkdtemp()


@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username


@app.route("/download_transcription_results/<path:file_name>", methods=["GET"])
@auth.login_required
def download_transcription_results(file_name):
    return send_from_directory(
        directory=transcription_results_temp_dir,
        path=file_name,
        as_attachment=True,
    )


@app.route("/<string:task_id>", methods=["GET"])
@auth.login_required
def get_transcription_results(task_id):    
    def _build_confidence_segments(segment: dict) -> List[dict]:
        confidence_segments = []
        current_confidence_segment_words = []
        current_confidence_segment_class = None

        for word in segment["words"]:
            if word["score"] >= 0.6:
                word_confidence_class = "high"
            elif word["score"] >= 0.2:
                word_confidence_class = "normal"
            else:
                word_confidence_class = "low"
        
            if current_confidence_segment_class is None:
                current_confidence_segment_class = word_confidence_class

            if current_confidence_segment_class == word_confidence_class:
                current_confidence_segment_words.append(word)
            else:
                confidence_segments.append({
                    "words": current_confidence_segment_words,
                    "confidence_class": current_confidence_segment_class,
                })
                current_confidence_segment_words = [word]
                current_confidence_segment_class = word_confidence_class
        
        if len(current_confidence_segment_words) > 0:
            confidence_segments.append({
                "words": current_confidence_segment_words,
                "confidence_class": current_confidence_segment_class,
            })
        
        return confidence_segments

    # Check if the results exist for this UUID
    blob_name = f"{task_id}.json"
    blob = results_bucket.blob(blob_name)
    
    if blob.exists():
        # Download the blob to a temporary file
        output_file_path = os.path.join(transcription_results_temp_dir, f"{task_id}.json")
        blob.download_to_filename(output_file_path)

        # Create a list of pre-formatted transcription results based on the JSON
        with open(output_file_path, "r", encoding="utf-8") as f:
            transcription_task_result_json = json.load(f)

        # Build confidence segments (segments with same confidence level between words)
        # for visualisation purposes
        for _, results_json in transcription_task_result_json["results"].items():
            for segment in results_json["segments"]:
                segment["confidence_segments"] = _build_confidence_segments(segment)

        # Send the file for download
        return render_template(
            "results.html",
            transcription_results=transcription_task_result_json["results"],
            temp_dir=transcription_results_temp_dir,
            file_name=f"{task_id}.json",
        )
    else:
        return render_template("no_results.html")


@app.route("/", methods=["GET", "POST"])
@auth.login_required
def index():
    if request.method == "POST":
        task_id = str(uuid.uuid4())
        task_type = request.form.get("task", "transcription")
        email = request.form.get("email")
        audio_files = request.files.getlist("audio_file")

        # If there are no audio files to process
        if len(audio_files) == 0:
            return redirect("/")
        
        # Create a config JSON file and upload the config JSON and audio files
        # to the ingestion DB under the task ID
        config = {
            "task_type": task_type,
            "email": email,
        }

        config_blob = ingestion_bucket.blob(f"{task_id}/config.json")
        config_blob.upload_from_string(json.dumps(config, ensure_ascii=False))

        for audio_file in audio_files:
            blob = ingestion_bucket.blob(f"{task_id}/{audio_file.filename}")
            blob.upload_from_file(audio_file)
        
        return redirect(f"/{task_id}")
    
    return render_template("index.html")


# Custom filter to format timestamps
@app.template_filter('format_segment_timestamp')
def format_segment_timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=_DEBUG, use_reloader=_USE_RELOADER)
