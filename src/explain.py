import numpy as np
from lime.lime_text import LimeTextExplainer
from .predict import predict, predict_batch, LABELS
from .preprocess import clean_telugu_text, find_trigger_words

# One shared explainer — class_names must match LABELS order {0,1,2}
explainer = LimeTextExplainer(class_names=[LABELS[i] for i in sorted(LABELS)])


# ---------------------------------------------------------------------------
# LIME classifier function — batched for speed
# ---------------------------------------------------------------------------

def _predict_proba_for_lime(texts: list) -> np.ndarray:
    """
    LIME requires: list[str] → np.ndarray of shape (n_samples, n_classes).
    Uses batched inference to avoid n*T4-overhead from sequential calls.
    """
    prob_dicts = predict_batch(texts)
    return np.array([
        [d["Real"], d["Fake"], d["Unverifiable"]]
        for d in prob_dicts
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(text: str, num_features: int = 8, num_samples: int = 150):
    """
    Uses LIME to generate an HTML explanation showing which words most
    influenced the classification decision.

    Parameters
    ----------
    text : str
        Raw Telugu input text.
    num_features : int
        Top features (words) to highlight. Default raised to 8 for richer output.
    num_samples : int
        Number of perturbed samples LIME generates. Raised to 150 for accuracy;
        still fast thanks to batched inference.

    Returns
    -------
    explanation_html : str
        LIME-generated HTML suitable for Gradio's gr.HTML component.
    trigger_words : list[str]
        Fake-news trigger words found in the text (from curated list).
    """
    cleaned_txt = clean_telugu_text(text)
    trigger_words = find_trigger_words(cleaned_txt)

    if not cleaned_txt.strip():
        return "<p style='color:#888;'>No text to explain.</p>", []

    try:
        exp = explainer.explain_instance(
            cleaned_txt,
            _predict_proba_for_lime,
            num_features=num_features,
            num_samples=num_samples,
        )
        return exp.as_html(), trigger_words
    except Exception as e:
        return (
            f"<p style='color:#dc2626;'>Explanation generation failed: {str(e)}</p>",
            trigger_words,
        )
