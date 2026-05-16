"""
Telugu Fake News / Misinformation Classifier — Source Package
=============================================================
Exposes key public API for import convenience.
"""

# Fix PyTorch DLL error on Windows (must be before any torch imports)
import os
import sys
python_dir = os.path.dirname(sys.executable)
dll_paths = [
    python_dir,
    os.path.join(python_dir, 'Library', 'bin'),
    os.path.join(python_dir, 'DLLs'),
]
for path in dll_paths:
    if os.path.exists(path) and path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from .preprocess import clean_telugu_text, find_trigger_words, remove_stopwords
from .predict import predict, predict_batch
from .explain import generate_explanation
from .scraper import scrape_telugu_news
from .config import (
    BASE_MODEL_NAME, NUM_LABELS, LABELS, LABEL2ID, ID2LABEL,
    MAX_LENGTH, TARGET_F1, MODEL_DIR,
)

__all__ = [
    # Preprocessing
    "clean_telugu_text",
    "find_trigger_words",
    "remove_stopwords",
    # Inference
    "predict",
    "predict_batch",
    # Explainability
    "generate_explanation",
    # Data collection
    "scrape_telugu_news",
    # Config
    "BASE_MODEL_NAME",
    "NUM_LABELS",
    "LABELS",
    "LABEL2ID",
    "ID2LABEL",
    "MAX_LENGTH",
    "TARGET_F1",
    "MODEL_DIR",
]

__version__ = "1.0.0"
