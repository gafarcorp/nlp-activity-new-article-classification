"""Tests use tiny stand-in models trained on a toy corpus, so they check the plumbing of
the app (validation, hints, batch, errors), not the accuracy of the real models."""
import random

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.app.main import create_app
from backend.app.preprocessing import clean_text

VOCAB = {
    0: "government minister election treaty border president diplomats ceasefire talks",
    1: "game team season coach player match goal league championship overtime",
    2: "stocks profit shares market company earnings investors inflation interest",
    3: "software internet chip computer research scientists network quantum device",
}


def toy_corpus(n_per_class=60, seed=0):
    rnd = random.Random(seed)
    texts, labels = [], []
    for label, words in VOCAB.items():
        w = words.split()
        for _ in range(n_per_class):
            texts.append(" ".join(rnd.choices(w, k=8)))
            labels.append(label)
    return texts, labels


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    texts, labels = toy_corpus()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([clean_text(t) for t in texts])
    joblib.dump(vec, d / "tfidf_vectorizer.joblib")
    joblib.dump(LogisticRegression(max_iter=500, random_state=0).fit(X, labels), d / "best_model_logreg.joblib")
    joblib.dump(RandomForestClassifier(n_estimators=20, random_state=0).fit(X, labels), d / "random_forest.joblib")
    return d


@pytest.fixture(scope="session")
def client(models_dir):
    with TestClient(create_app(models_dir)) as c:
        yield c


@pytest.fixture()
def empty_client(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        yield c
