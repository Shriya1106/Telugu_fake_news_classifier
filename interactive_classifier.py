"""
Interactive Telugu Fake News Classifier
========================================
Type your own Telugu text and get instant results!
"""

import sys
import os

# Fix PyTorch DLL error on Windows
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

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict
from src.preprocess import find_trigger_words, clean_telugu_text

print("=" * 70)
print("TELUGU FAKE NEWS CLASSIFIER - INTERACTIVE MODE")
print("=" * 70)
print()
print("Loading model... (this takes a few seconds)")
print()

# Pre-load model
from src.predict import _load_model
_load_model()

print("✅ Model loaded successfully!")
print()
print("=" * 70)
print()

while True:
    print("Enter Telugu text to analyze (or 'quit' to exit):")
    print()
    
    # Get input
    text = input("> ")
    
    if text.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    if not text.strip():
        print("\n⚠️  Please enter some text!\n")
        continue
    
    print("\n" + "=" * 70)
    print("ANALYZING...")
    print("=" * 70)
    
    try:
        # Get prediction
        label, confidence, prob_dict = predict(text)
        
        # Display results
        print(f"\n✅ RESULT:")
        print(f"\n  Verdict: {label}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"\n  Probability Distribution:")
        for lbl, prob in prob_dict.items():
            bar = "█" * int(prob * 50)
            print(f"    {lbl:12s}: {prob*100:5.1f}% {bar}")
        
        # Check trigger words
        cleaned = clean_telugu_text(text)
        triggers = find_trigger_words(cleaned)
        if triggers:
            print(f"\n  ⚠️  Suspicious Keywords: {', '.join(triggers)}")
        else:
            print(f"\n  ✅ No suspicious keywords")
        
        print("\n" + "=" * 70)
        print()
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")

print()
