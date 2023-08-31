import torch
import whisperx
import gc 
import time
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Transcript application.")

    parser.add_argument(
        "--audio-path", 
        type=str,
        required=False,
        default="test-audio-128kbps.mp3", 
        help="Path to an audio file.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["small", "medium", "large-v1", "large-v2"],
        required=False,
        default="large-v2", 
        help="Whisper model name.")
    parser.add_argument(
        "--hf-auth-token", 
        type=str, 
        required=False,
        default=None, 
        help="Auth token for the HuggingFace API.")
    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["float16", "int8"],
        required=False,
        default="float16",
        help="Model weight dtype.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=16,
        help="Batch size for batched inference",
    )
    args = parser.parse_args()

    cuda_is_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    print(f"Torch CUDA is available: {cuda_is_available}, device count: {cuda_device_count}")
    device = "cuda" if cuda_is_available else "cpu"

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=args.compute_type)

    print(f"Reading audio file from: {args.audio_path}")
    audio = whisperx.load_audio(args.audio_path)
    print(f"Audio data: {audio}")

    execution_times = {}

    s_time_1 = time.time()
    result = model.transcribe(audio, batch_size=args.batch_size)
    t_time_1 = time.time() - s_time_1
    execution_times["transcription"] = t_time_1
    print(result["segments"]) # before alignment
    print(f"Transcription took: {t_time_1} sec")

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    s_time_2 = time.time()
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    t_time_2 = time.time() - s_time_2
    execution_times["output alignment"] = t_time_2
    print(result["segments"]) # after alignment
    print(f"Whisper output alignment took: {t_time_2} sec")

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Diarize and assign speaker labels
    if args.hf_auth_token:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=args.hf_auth_token, device=device)

        # add min/max number of speakers if known
        s_time_3 = time.time()
        diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        t_time_3 = time.time() - s_time_3
        execution_times["diarization"] = t_time_3
        print(f"Diarization took: {t_time_3} sec")

        s_time_4 = time.time()
        result = whisperx.assign_word_speakers(diarize_segments, result)
        t_time_4 = time.time() - s_time_4
        execution_times["word speaker alignment"] = t_time_4
        print(f"Word speaker assignment took: {t_time_4} sec")

        print(diarize_segments)
        print(result["segments"]) # segments are now assigned speaker IDs

    print(f"Transcription took in total: {sum(execution_times.values())} sec, breakdown: {execution_times}")


if __name__ == "__main__":
    main()