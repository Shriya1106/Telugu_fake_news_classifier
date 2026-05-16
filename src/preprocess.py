import re

# --- Telugu Stopwords (common particles, postpositions, conjunctions) ---
TELUGU_STOPWORDS = {
    "మరియు", "కానీ", "లేదా", "కూడా", "అయితే", "ఈ", "ఆ", "ఒక",
    "ఇది", "అది", "వారు", "మేము", "నేను", "మీరు", "అతను", "ఆమె",
    "ఇక్కడ", "అక్కడ", "ఎక్కడ", "ఎప్పుడు", "ఎలా", "ఏమి", "ఎవరు",
    "చాలా", "అన్ని", "కొన్ని", "ప్రతి", "తరువాత", "ముందు", "మధ్య",
    "పైన", "కింద", "లో", "కి", "తో", "నుండి", "కోసం", "వల్ల",
    "గా", "గురించి", "ద్వారా", "వరకు", "లేదు", "ఉంది", "ఉన్నారు",
    "ఉన్నాయి", "అయినా", "అయిన", "చేసిన", "చేసే", "అని", "అనే",
    "మాత్రమే", "కాదు", "హా", "ఓ", "నా", "మా", "తన", "వాళ్ళ"
}

# --- Fake news trigger words (urgency, emotional manipulation, unverified claims) ---
TELUGU_TRIGGER_WORDS = {
    "వెంటనే", "షేర్", "ఫార్వర్డ్", "ఉచితంగా", "ఫ్రీ", "లింక్",
    "క్లిక్", "తప్పనిసరిగా", "100%", "గ్యారంటీ", "నిజం",
    "నమ్మండి", "షాకింగ్", "బ్రేకింగ్", "వైరల్", "సంచలనం",
    "రహస్యం", "దయచేసి", "అత్యవసరం", "ప్రమాదం", "హెచ్చరిక",
    "నిషేధం", "చివరి అవకాశం", "ఇప్పుడే", "తగ్గిపోతుంది",
    "నయమవుతుంది", "రూపాయలు", "డబ్బు", "బహుమతి", "లాటరీ"
}


def is_telugu(text: str) -> bool:
    """
    Check if text contains Telugu Unicode characters (U+0C00-U+0C7F).

    This is the single canonical implementation — other modules should
    import this instead of re-implementing the check.

    Args:
        text: Input text string

    Returns:
        True if text contains at least one Telugu character, False otherwise
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[\u0C00-\u0C7F]", text))


def clean_telugu_text(text: str) -> str:
    """
    Preprocesses Telugu text for NLP models.
    Steps:
      1. Remove URLs
      2. Remove emojis and non-standard Unicode symbols
      3. Remove special characters (keep Telugu + basic punctuation + digits)
      4. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 2. Remove emojis and other non-BMP characters
    text = re.sub(
        r'[\U00010000-\U0010ffff]', '', text, flags=re.UNICODE
    )

    # 3. Keep Telugu Unicode block (U+0C00-U+0C7F), digits, whitespace, basic punctuation
    text = re.sub(r'[^\u0C00-\u0C7F\w\s.,!?0-9\-]', '', text)

    # 4. Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """Removes Telugu stopwords from text."""
    words = text.split()
    return " ".join(w for w in words if w not in TELUGU_STOPWORDS)


def find_trigger_words(text: str) -> list:
    """Returns a list of fake-news trigger words found in the given text."""
    words = text.split()
    return [w for w in words if w in TELUGU_TRIGGER_WORDS]


if __name__ == "__main__":
    sample = "ప్రభుత్వం రైతులందరికీ రూ. 10,000 ఉచితంగా ఇస్తోంది. వెంటనే ఈ లింక్ క్లిక్ చేయండి! https://fake.url.com 🚀"
    cleaned = clean_telugu_text(sample)
    print(f"Cleaned : {cleaned}")
    print(f"Triggers: {find_trigger_words(cleaned)}")
    print(f"No stops: {remove_stopwords(cleaned)}")
    print(f"Is Telugu: {is_telugu(sample)}")
