# Fix PyTorch DLL error on Windows (must be before torch import)
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

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .preprocess import clean_telugu_text
from .config import BASE_MODEL_NAME, NUM_LABELS, LABELS, MODEL_DIR, MAX_LENGTH
from typing import List, Tuple, Dict, Union

# ---------------------------------------------------------------------------
# Lazy-loaded globals (avoids downloading ~500 MB at import time)
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None


def _load_model():
    """Load the model and tokenizer once, on first call."""
    global _tokenizer, _model
    if _model is not None:
        return

    print("Loading Model & Tokenizer for inference...")

    # Try the absolute path from config first, then relative fallback
    model_candidates = [
        MODEL_DIR,                        # Absolute path from config.py
        "./telugu-fake-news-model",       # Relative CWD fallback
    ]

    loaded = False
    for model_path in model_candidates:
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            try:
                _tokenizer = AutoTokenizer.from_pretrained(model_path)
                _model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=NUM_LABELS
                )
                print(f"✅ Loaded fine-tuned model from '{model_path}'.")
                loaded = True
                break
            except Exception as e:
                print(f"⚠️  Failed to load from '{model_path}': {e}")

    if not loaded:
        # Fallback to base model from Hugging Face
        print(
            f"⚠️  Fine-tuned model not found at any expected location. "
            f"Falling back to base model '{BASE_MODEL_NAME}' for demo."
        )
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=NUM_LABELS
        )

    _model.eval()


def get_model_and_tokenizer():
    """Public accessor for the loaded model and tokenizer.

    Returns
    -------
    tokenizer, model : tuple
        The loaded tokenizer and model objects.
    """
    _load_model()
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Single-text inference
# ---------------------------------------------------------------------------

def predict(text: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Classify a Telugu text as Real / Fake / Unverifiable.

    Returns
    -------
    label : str          — "Real", "Fake", or "Unverifiable"
    confidence : float   — probability of the predicted class
    prob_dict : dict      — {"Real": p0, "Fake": p1, "Unverifiable": p2}
    """
    _load_model()

    cleaned_text = clean_telugu_text(text)
    if not cleaned_text.strip():
        return "Unknown", 0.0, {"Real": 0.0, "Fake": 0.0, "Unverifiable": 0.0}

    inputs = _tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    logits = outputs.logits                          # shape: (1, NUM_LABELS)
    probs = F.softmax(logits, dim=1).squeeze().tolist()

    # Guard: softmax on single logit returns a float, not a list
    if isinstance(probs, float):
        probs = [probs]
    while len(probs) < 3:
        probs.append(0.0)

    pred_idx = torch.argmax(logits, dim=1).item()
    if pred_idx not in LABELS:
        pred_idx = 0

    predicted_label = LABELS[pred_idx]
    confidence = probs[pred_idx]

    prob_dict = {
        "Real":         round(probs[0], 4),
        "Fake":         round(probs[1], 4),
        "Unverifiable": round(probs[2], 4),
    }

    return predicted_label, confidence, prob_dict


# ---------------------------------------------------------------------------
# Batched inference — used by LIME to avoid per-sample overhead
# ---------------------------------------------------------------------------

def predict_batch(texts: List[str]) -> List[Dict[str, float]]:
    """
    Classify a list of texts in a single forward pass.

    Returns
    -------
    list of prob_dicts: [{"Real": p0, "Fake": p1, "Unverifiable": p2}, ...]
    """
    _load_model()

    if not texts:
        return []

    cleaned = [clean_telugu_text(t) for t in texts]
    # Replace empty strings with a single space so the tokenizer doesn't error
    cleaned = [t if t.strip() else " " for t in cleaned]

    # Process in smaller batches to avoid OOM on CPU
    BATCH_SIZE = 32
    all_results = []

    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i : i + BATCH_SIZE]

        inputs = _tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = _model(**inputs)

        probs_batch = F.softmax(outputs.logits, dim=1).tolist()

        for probs in probs_batch:
            while len(probs) < 3:
                probs.append(0.0)
            all_results.append({
                "Real":         round(probs[0], 4),
                "Fake":         round(probs[1], 4),
                "Unverifiable": round(probs[2], 4),
            })

    return all_results
