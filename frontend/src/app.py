from flask import Flask, render_template, request, redirect
from google.cloud import storage

app = Flask(__name__)

# Initialize a Cloud Storage client.
storage_client = storage.Client()
bucket_name = "your-bucket-name"
bucket = storage_client.get_bucket(bucket_name)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio_file = request.files["audio_file"]
        blob = bucket.blob(audio_file.filename)
        blob.upload_from_file(audio_file)
        return redirect("/")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
