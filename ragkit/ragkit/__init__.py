"""ragkit: local retrieval-augmented generation over your own documents.

No API keys, no server, no network on the default path. Python
stdlib only.
"""

from .answer import (
    Answer,
    Answerer,
    Citation,
    ExtractiveAnswerer,
    OpenAICompatibleAnswerer,
)
from .chunk import Chunk, split_text
from .eval import EvalSummary, GoldenQuestion, load_golden, report_text, run_eval
from .ingest import load_documents, load_index, save_index
from .retrieve import Retriever, ScoredChunk, select_for_answer, top_relative

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Answerer",
    "Citation",
    "Chunk",
    "EvalSummary",
    "ExtractiveAnswerer",
    "GoldenQuestion",
    "OpenAICompatibleAnswerer",
    "Retriever",
    "ScoredChunk",
    "load_documents",
    "load_golden",
    "load_index",
    "report_text",
    "run_eval",
    "save_index",
    "select_for_answer",
    "split_text",
    "top_relative",
]
