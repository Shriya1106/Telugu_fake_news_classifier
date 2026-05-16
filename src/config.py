"""
Telugu Fake News Classifier — Centralized Configuration
========================================================
All model, training, and deployment constants in one place.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "telugu-fake-news-model")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "google/muril-base-cased"
NUM_LABELS = 3
LABELS = {0: "Real", 1: "Fake", 2: "Unverifiable"}
LABEL2ID = {"Real": 0, "Fake": 1, "Unverifiable": 2}
ID2LABEL = {0: "Real", 1: "Fake", 2: "Unverifiable"}   # int → label name
MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TARGET_F1 = 0.76

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.6  # Below this, flag as low-confidence

# ---------------------------------------------------------------------------
# LIME Explainability
# ---------------------------------------------------------------------------
LIME_NUM_FEATURES = 8
LIME_NUM_SAMPLES = 150

# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
TELUGU_RSS_FEEDS = {
    "Sakshi": "https://www.sakshi.com/rss/telangana",
    "Eenadu": "https://www.eenadu.net/telangana/rss",
    "TV9 Telugu": "https://tv9telugu.com/feed",
}
MAX_ARTICLES_PER_SOURCE = 20
