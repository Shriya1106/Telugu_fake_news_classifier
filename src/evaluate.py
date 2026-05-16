"""
Telugu Fake News Classifier — Standalone Evaluation Script
===========================================================
Evaluates a trained model on a test set and prints detailed metrics.

Usage:
    python -m src.evaluate                          # uses built-in test set
    python -m src.evaluate --csv data/test.csv      # uses a custom CSV (text, label columns)
"""

import argparse
import sys
import os

# Fix UnicodeEncodeError in Windows terminal for Telugu and Emojis
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from .predict import predict, get_model_and_tokenizer
from .config import LABELS, TARGET_F1

# ---------------------------------------------------------------------------
# Built-in test samples (covers all 3 classes)
# ---------------------------------------------------------------------------

TEST_SAMPLES = [
    # --- Real (label 0) ---
    ("హైదరాబాద్‌లో నిన్న భారీ వర్షం కారణంగా పలు ప్రాంతాలు జలమయం అయ్యాయి.", 0),
    ("భారత క్రికెట్ జట్టు ఆస్ట్రేలియాపై 5 వికెట్ల తేడాతో గెలిచింది.", 0),
    ("తెలంగాణ రాష్ట్రంలో కొత్త ఐటీ పార్క్ ప్రారంభం అయింది.", 0),
    ("కేంద్ర బడ్జెట్‌లో విద్యా రంగానికి అధిక నిధులు కేటాయించారు.", 0),
    ("భారత ఆర్థిక వ్యవస్థ 7.2 శాతం వృద్ధి సాధించింది.", 0),
    ("హైదరాబాద్ మెట్రో రైల్ కొత్త మార్గంలో ప్రయాణం ప్రారంభం.", 0),
    ("ఆంధ్రప్రదేశ్ ప్రభుత్వం కొత్త విద్యా విధానాన్ని ప్రకటించింది.", 0),
    ("రేపు ఉదయం 10 గంటలకు ఇస్రో కొత్త ఉపగ్రహాన్ని ప్రయోగించనుంది.", 0),

    # --- Fake (label 1) ---
    ("ఈ లింక్ క్లిక్ చేస్తే ఉచితంగా జియో ఫోన్ వస్తుంది. ఫార్వర్డ్ చేయండి!", 1),
    ("ప్రధానమంత్రి ప్రతి పౌరుడికి రూ. 15 లక్షలు ఇస్తారు. ఇప్పుడే అప్లై చేయండి!", 1),
    ("ఈ మెసేజ్ 10 మందికి ఫార్వర్డ్ చేస్తే రూ. 1000 వస్తుంది. నిజం!", 1),
    ("రేపటి నుండి పెట్రోల్ ధర లీటర్‌కు రూ. 30 తగ్గించారు! షేర్ చేయండి!", 1),
    ("NASA ప్రకటించింది: రేపు 3 గంటలు భూమిపై చీకటి అవుతుంది!", 1),
    ("ఈ వాట్సాప్ మెసేజ్ డిలీట్ చేస్తే మీ ఫోన్ హ్యాక్ అవుతుంది!", 1),
    ("గూగుల్ ప్రతి భారతీయుడికి రూ. 25,000 ఇస్తోంది. లింక్ క్లిక్ చేయండి!", 1),
    ("భారత ఎన్నికలు రద్దు అయ్యాయి! బ్రేకింగ్ న్యూస్! షేర్ చేయండి!", 1),

    # --- Unverifiable (label 2) ---
    ("ఈ ఆకు రసం తాగితే ఏ జబ్బు అయినా 2 రోజుల్లో తగ్గిపోతుంది.", 2),
    ("ప్రముఖ నటుడు రహస్యంగా రాజకీయ పార్టీలో చేరారు.", 2),
    ("ఫోన్ రేడియేషన్ వల్ల తలనొప్పి వస్తుందని నిపుణులు చెప్తున్నారు.", 2),
    ("కొత్త వ్యాక్సిన్ పూర్తిగా సురక్షితం కాదని కొందరు వైద్యులు అంటున్నారు.", 2),
    ("ఈ పండు తింటే జ్ఞాపకశక్తి పెరుగుతుందని పరిశోధనలు చెప్తున్నాయి.", 2),
    ("ఆ రాజకీయ నేత పార్టీ మారనున్నారని వార్తలు వస్తున్నాయి.", 2),
    ("ఈ తేయాకు తాగితే బరువు తగ్గుతుందని సోషల్ మీడియాలో ప్రచారం.", 2),
    ("వచ్చే సంవత్సరం భారీ కరువు వస్తుందని వాతావరణ నిపుణులు.", 2),
]


def evaluate_builtin():
    """Evaluate the model on the built-in test set."""
    print("=" * 60)
    print("📊 Telugu Fake News Classifier — Evaluation")
    print("=" * 60)

    y_true = []
    y_pred = []
    results_detail = []

    label_to_id = {v: k for k, v in LABELS.items()}

    for text, true_label in TEST_SAMPLES:
        pred_label, confidence, prob_dict = predict(text)
        pred_id = label_to_id.get(pred_label, 0)
        y_true.append(true_label)
        y_pred.append(pred_id)
        results_detail.append({
            "text": text[:50] + "…",
            "true": LABELS[true_label],
            "pred": pred_label,
            "conf": f"{confidence:.2%}",
            "correct": "✅" if true_label == pred_id else "❌",
        })

    # Print per-sample results
    print(f"\n{'Text':<55} {'True':<15} {'Pred':<15} {'Conf':<8} {'OK'}")
    print("-" * 100)
    for r in results_detail:
        print(f"{r['text']:<55} {r['true']:<15} {r['pred']:<15} {r['conf']:<8} {r['correct']}")

    # Aggregate metrics
    target_names = [LABELS[i] for i in sorted(LABELS)]
    print("\n" + "=" * 60)
    print("📈 Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("📊 Confusion Matrix")
    print(f"{'':>15} {'Real':>10} {'Fake':>10} {'Unver.':>10}")
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>15} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    # Overall metrics
    f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Weighted F1 : {f1:.4f}  (Target: > {TARGET_F1})")
    print(f"🎯 Accuracy    : {acc:.4f}")

    if f1 > TARGET_F1:
        print("🎉 SUCCESS — F1 target exceeded!")
    else:
        print("⚠️  F1 below target. Fine-tune with more data or epochs.")

    return f1, acc


def evaluate_csv(csv_path: str):
    """Evaluate the model on a custom CSV file with 'text' and 'label' columns."""
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]

    print(f"📂 Loaded {len(df)} samples from {csv_path}")
    print(f"   Text column: '{text_col}', Label column: '{label_col}'")

    label_to_id = {v: k for k, v in LABELS.items()}

    y_true = df[label_col].tolist()
    y_pred = []

    for text in df[text_col]:
        pred_label, _, _ = predict(str(text))
        pred_id = label_to_id.get(pred_label, 2)
        y_pred.append(pred_id)

    target_names = [LABELS[i] for i in sorted(LABELS)]
    print("\n" + classification_report(y_true, y_pred, target_names=target_names, digits=4))

    f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    print(f"🎯 Weighted F1 : {f1:.4f}")
    print(f"🎯 Accuracy    : {acc:.4f}")
    return f1, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Telugu Fake News Classifier")
    parser.add_argument("--csv", type=str, default=None, help="Path to test CSV (text, label)")
    args = parser.parse_args()

    if args.csv:
        evaluate_csv(args.csv)
    else:
        evaluate_builtin()
