from typing import Dict, Any

from models import TranscriptionResult, TranscriptionSegment, TranscriptionWordSegment
from logger import get_logger


logger = get_logger(__name__)


def dict_to_transcription_word_segment(
    transcription_word_dict: Dict[str, Any],
) -> TranscriptionWordSegment:
    return TranscriptionWordSegment(**transcription_word_dict)


def dict_to_transcription_segment(
    transcription_segment_dict: Dict[str, Any],
) -> TranscriptionSegment:
    words = []

    for word in transcription_segment_dict["words"]:
        try:
            words.append(dict_to_transcription_word_segment(word))
        except Exception as ex:
            logger.error(f"Failed to convert dict to TranscriptionWordSegment: {ex}, dict: {word}")

    return TranscriptionSegment(
        start=transcription_segment_dict["start"],
        end=transcription_segment_dict["end"],
        text=transcription_segment_dict["text"].strip(),
        words=words,
        speaker=transcription_segment_dict.get("speaker"),
    )


def dict_to_transcription_result(transcription_result: Dict[str, Any]) -> TranscriptionResult:
    segments = []

    for segment in transcription_result["segments"]:
        try:
            segments.append(dict_to_transcription_segment(segment))
        except Exception as ex:
            logger.error(f"Failed to convert dict to TranscriptionSegment: {ex}, dict: {segment}")

    word_segments = []

    for word_segment in word_segments:
        try:
            word_segments.append(dict_to_transcription_word_segment(word_segment))
        except Exception as ex:
            logger.error(f"Failed to convert dict to TranscriptionWordSegment: {ex}, dict: {word_segment}")

    return TranscriptionResult(
        segments=segments,
        word_segments=word_segments,
        execution_times=transcription_result["execution_times"],
    )
