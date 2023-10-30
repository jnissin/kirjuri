from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class TranscriptionWordSegment:
    word: str
    start: float
    end: float
    score: float
    speaker: Optional[str]


@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float
    words: List[TranscriptionWordSegment]
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    segments: List[TranscriptionSegment]
    word_segments: List[TranscriptionWordSegment]
    execution_times: Dict[str, float]


@dataclass
class TaskTranscriptionResult:
    task_id: str
    results: Dict[str, TranscriptionResult]