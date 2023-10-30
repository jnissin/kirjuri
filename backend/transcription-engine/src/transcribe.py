import os
import time
import json
import torch
import whisperx
import argparse

from dataclasses import asdict, dataclass
from typing import Any, List, Dict, Optional, Tuple
from google.cloud import storage
from whisperx.asr import FasterWhisperPipeline
from whisperx import DiarizationPipeline
from transformers import Wav2Vec2ForCTC

from logger import get_logger
from models import TaskTranscriptionResult, TranscriptionResult
from parsing import dict_to_transcription_result
from integrations.email import EmailClient, MailjetEmailClient
from settings import DOMAIN_NAME, FROM_EMAIL
from align_patch import align_patched


logger = get_logger(__name__)


@dataclass
class TranscriptionContext:
    whisper_model: FasterWhisperPipeline
    diarize_model: DiarizationPipeline
    device: str
    batch_size: int
    _align_model_cache: Optional[dict] = None

    def get_align_model(self, language: str) -> Tuple[Wav2Vec2ForCTC, Dict[str, Any]]:
        if self._align_model_cache is None:
            self._align_model_cache = {}

        if language not in self._align_model_cache:
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=self.device
            )
            self._align_model_cache[language] = align_model, metadata
        return self._align_model_cache.get(language)


def send_transcription_results_notification_email(
    email_client: EmailClient,
    to_email: str,
    transcription_task_id: str,
):
    transcription_results_url = f"https://{DOMAIN_NAME}/{transcription_task_id}"
    email_content = f"""
<h3>Your transcriptions have been created!</h3>
<p>
The transcriptions for task ID: {transcription_task_id} are complete.
</p>
<p>
You can view the transcriptions results at: <a href="{transcription_results_url}">{transcription_results_url}</a>
</p>
"""
    email_client.send_email(
        from_email=FROM_EMAIL,
        to_email=to_email,
        subject="Your transcriptions are ready",
        content=email_content,
        from_name=None,
        to_name=None,
    )


def transcribe(
    transcription_context: TranscriptionContext, audio_file_path: str, task_type: str
) -> dict:
    logger.info(f"Transcribing {audio_file_path} with task type {task_type}")

    logger.info(f"Reading audio file from: {audio_file_path}")
    audio = whisperx.load_audio(audio_file_path)
    execution_times = {}

    # 1. Transcribe with original whisper (batched)
    s_time_1 = time.time()
    result = transcription_context.whisper_model.transcribe(
        audio, batch_size=transcription_context.batch_size
    )
    t_time_1 = time.time() - s_time_1
    execution_times["whisper_transcription"] = t_time_1
    logger.info(f"Whisper transcription took: {t_time_1} sec")

    # 2. Align whisper output
    model_a, metadata = transcription_context.get_align_model(
        language=result["language"]
    )
    s_time_2 = time.time()
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        transcription_context.device,
        return_char_alignments=False,
    )
    t_time_2 = time.time() - s_time_2
    execution_times["output_alignment"] = t_time_2
    logger.info(f"Whisper output alignment took: {t_time_2} sec")

    if task_type == "diarization" and transcription_context.diarize_model:
        # add min/max number of speakers if known
        # 3. Diarize segments
        s_time_3 = time.time()
        diarize_segments = transcription_context.diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        t_time_3 = time.time() - s_time_3
        execution_times["diarization"] = t_time_3
        logger.info(f"Diarization took: {t_time_3} sec")

        # 4. Assign per word speaker IDs
        s_time_4 = time.time()
        result = whisperx.assign_word_speakers(diarize_segments, result)
        t_time_4 = time.time() - s_time_4
        execution_times["word_speaker_alignment"] = t_time_4
        logger.info(f"Word speaker assignment took: {t_time_4} sec")

    result["execution_times"] = execution_times
    return result


def read_and_transcribe_from_bucket(
    transcription_context: TranscriptionContext,
    ingestion_bucket_name: str,
    results_bucket_name: str,
    email_client: Optional[EmailClient],
) -> List[TranscriptionResult]:
    # List all blobs in the ingestion bucket
    storage_client = storage.Client()
    ingestion_bucket = storage_client.get_bucket(ingestion_bucket_name)
    results_bucket = storage_client.get_bucket(results_bucket_name)
    blobs = ingestion_bucket.list_blobs()

    # Process audio files for each transcription task ID one ID at a time
    processed_task_ids = set()
    num_processed_audio_files = 0
    task_transcription_results = []

    for blob in blobs:
        # Extract transcription task ID from the blob name
        transcription_task_id = blob.name.split("/")[0]

        # Skip blobs belonging to task IDs that we have already processed
        if transcription_task_id in processed_task_ids:
            continue

        transcription_results = {}

        # Read and parse the config.json file for this task ID
        config_blob = ingestion_bucket.blob(f"{transcription_task_id}/config.json")
        config_json = json.loads(config_blob.download_as_text())
        task_type = config_json.get("task_type", "transcription")
        email = config_json.get("email", None)

        # Get all blobs with this task ID (excluding config.json) i.e. all audio files
        audio_blobs = ingestion_bucket.list_blobs(prefix=f"{transcription_task_id}/")
        audio_blobs = [ab for ab in audio_blobs if "config.json" not in ab.name]

        for audio_blob in audio_blobs:
            # Call the transcribe method for each audio file
            audio_file_path = f"/tmp/{audio_blob.name.replace('/', '_')}"
            audio_blob.download_to_filename(audio_file_path)
            transcription_result = transcribe(
                transcription_context=transcription_context,
                audio_file_path=audio_file_path,
                task_type=task_type,
            )

            # Remove task ID from blob name
            audio_file_name = os.path.basename(audio_file_path).replace(
                transcription_task_id, ""
            )
            audio_file_name = (
                audio_file_name[1:]
                if audio_file_name.startswith("_")
                else audio_file_name
            )
            transcription_results[audio_file_name] = dict_to_transcription_result(
                transcription_result
            )
            num_processed_audio_files += 1

        # Create result object and write to GCS
        task_transcription_result = TaskTranscriptionResult(
            task_id=transcription_task_id,
            results=transcription_results,
        )
        task_transcription_results.append(task_transcription_result)

        config_blob = results_bucket.blob(f"{transcription_task_id}.json")
        config_blob.upload_from_string(
            json.dumps(asdict(task_transcription_result), ensure_ascii=False)
        )

        # Delete all blobs (audio files and config.json) for this task ID
        for blob_to_delete in ingestion_bucket.list_blobs(
            prefix=f"{transcription_task_id}/"
        ):
            blob_to_delete.delete()

        # Mark the task ID as processed
        processed_task_ids.add(transcription_task_id)

        # Notify user via email
        if email:
            if email_client:
                send_transcription_results_notification_email(
                    email_client=email_client,
                    to_email=email,
                    transcription_task_id=transcription_task_id,
                )
            else:
                logger.warning(f"Could not send notification to email {email} because there is no email client")

    logger.info(
        f"Successfully processed: {len(processed_task_ids)} tasks with a total of {num_processed_audio_files} audio files"
    )
    return task_transcription_results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Transcript application.")

    parser.add_argument(
        "--ingestion-bucket",
        type=str,
        required=False,
        default="kirjuri-ingestion-db",
        help="The name of the ingestion GCS bucket",
    )
    parser.add_argument(
        "--results-bucket",
        type=str,
        required=False,
        default="kirjuri-results-db",
        help="The name of the results GCS bucket",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "medium", "large-v1", "large-v2"],
        required=False,
        default="large-v2",
        help="Whisper model name",
    )
    parser.add_argument(
        "--hf-auth-token",
        type=str,
        required=False,
        default=None,
        help="Auth token for the HuggingFace API",
    )
    parser.add_argument(
        "--mailjet-api-key",
        type=str,
        required=False,
        default=None,
        help="API key for the Mailjet API",
    )
    parser.add_argument(
        "--mailjet-secret-key",
        type=str,
        required=False,
        default=None,
        help="Secret key for the Mailjet API",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["float16", "int8"],
        required=False,
        default="float16",
        help="Model weight dtype",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=16,
        help="Batch size for batched inference",
    )
    args = parser.parse_args()

    # Apply WhisperX align method patch
    logger.info(
        f"Applying align method patch to WhisperX library .."
    )
    whisperx.align = align_patched
    whisperx.alignment.align = align_patched

    cuda_is_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    device = "cuda" if cuda_is_available else "cpu"
    logger.info(
        f"Torch CUDA is available: {cuda_is_available}, device count: {cuda_device_count}"
    )

    # Pre-load models
    logger.info(f"Loading whisper model: {args.model} ..")
    whisper_model = whisperx.load_model(
        args.model, device, compute_type=args.compute_type
    )

    if args.hf_auth_token:
        logger.info("Loading diarization model ..")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=args.hf_auth_token, device=device)
    else:
        logger.warning("Diarization disabled: No HF auth token")

    logger.info("Creating transcription context ..")
    transcription_context = TranscriptionContext(
        whisper_model=whisper_model,
        diarize_model=diarize_model,
        device=device,
        batch_size=args.batch_size,
    )

    if args.mailjet_api_key and args.mailjet_secret_key:
        logger.info("Creating email client ..")
        email_client = MailjetEmailClient(
            api_key=args.mailjet_api_key,
            secret_key=args.mailjet_secret_key,
        )
    else:
        logger.warning("Email sending disabled: No API keys")
        email_client = None

    s_time = time.time()
    logger.info("Starting transcription for GCS audio files ..")
    read_and_transcribe_from_bucket(
        transcription_context=transcription_context,
        ingestion_bucket_name=args.ingestion_bucket,
        results_bucket_name=args.results_bucket,
        email_client=email_client,
    )
    t_time = time.time() - s_time
    logger.info(f"Transcription took in total: {t_time} sec")


if __name__ == "__main__":
    main()
