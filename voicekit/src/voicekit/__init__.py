"""voicekit: a small, fully offline voice-command pipeline built from scratch on NumPy."""
from .intents import Intent, group_events, parse_events, parse_words
from .models import MLP, DTWClassifier, dtw_distance
from .pipeline import CLASSES, WORDS, Event, Recognizer, build_dataset, calibrate_thresholds, render_stream, score_events, train_model
from .wake import WakeGate
from .assistant import Assistant, Command
from .segmenter import Segment, SegmenterConfig, StreamSegmenter, segment_signal

__all__ = [
    "Intent", "group_events", "parse_events", "parse_words", "MLP", "DTWClassifier", "dtw_distance",
    "CLASSES", "WORDS", "Event", "Recognizer", "build_dataset", "calibrate_thresholds", "render_stream", "score_events",
    "train_model", "WakeGate", "Assistant", "Command", "Segment", "SegmenterConfig", "StreamSegmenter", "segment_signal",
]
__version__ = "0.1.0"
