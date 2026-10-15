"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # .../backend
MODELS_DIR = Path(os.getenv("NEWSSORT_MODELS_DIR", BASE_DIR / "models"))
FRONTEND_DIR = Path(os.getenv("NEWSSORT_FRONTEND_DIR", BASE_DIR.parent / "frontend"))
DATA_DIR = Path(__file__).resolve().parent / "data"

# Label order MUST match the notebook (label = class_index - 1).
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

MAX_CHARS = 2000            # title + description
MAX_BATCH_ROWS = 1000       # BERT on a CPU is slow; keep batches modest
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
BERT_MAX_LEN = 128          # same as training
BERT_BATCH_SIZE = 32

CLOSE_CALL_GAP = 0.15       # top-2 probabilities closer than this -> "close call"
OVERLAP_MIN_SECOND = 0.10   # Business/Sci-Tech note only if the runner-up is at least this likely

MODEL_FILES = {
    "vectorizer": "tfidf_vectorizer.joblib",
    "logreg": "best_model_logreg.joblib",
    "rf": "random_forest.joblib",
    "bert_dir": "bert_ag_news",
}
DISPLAY_NAMES = {"bert": "BERT", "logreg": "Logistic Regression", "rf": "Random Forest"}
