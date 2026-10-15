"""Text cleaning. These functions are copied from the training notebook and MUST stay
identical to it, otherwise the models see different text than they were trained on."""
import html
import re

import nltk


def _load_nltk():
    """Load stop words and the lemmatizer. Loading (not just looking for a folder) is the
    reliable check, because NLTK keeps some corpora zipped until first use."""
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("tests")
    return stop, lemmatizer


try:
    _STOP, _LEMMA = _load_nltk()
except LookupError:
    for _res in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(_res, quiet=True)
    try:
        _STOP, _LEMMA = _load_nltk()
    except LookupError as exc:  # fail loudly: silent fallbacks would change the features
        raise RuntimeError("NLTK data (stopwords, wordnet) is missing and could not be "
                           "downloaded. Run: python -m nltk.downloader stopwords wordnet omw-1.4") from exc


def fix_entities(text: str) -> str:
    text = html.unescape(str(text))
    # entities whose leading '&' was stripped in the AG News copy used for training
    text = re.sub(r"#39;", "'", text)
    text = re.sub(r"\bquot;", '"', text)
    text = re.sub(r"\bamp;", "&", text)
    text = re.sub(r"\blt;", "<", text)
    text = re.sub(r"\bgt;", ">", text)
    return text


def basic_clean(text: str) -> str:
    """Light cleaning used for BERT."""
    text = fix_entities(text)
    text = text.replace("\\", " ")
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    """Heavy cleaning used for the TF-IDF models (Logistic Regression, Random Forest)."""
    text = basic_clean(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in _STOP and len(t) > 1]
    return " ".join(_LEMMA.lemmatize(t) for t in tokens)
