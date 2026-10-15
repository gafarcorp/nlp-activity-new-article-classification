"""Loads the trained models once and turns texts into class probabilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from . import config
from .preprocessing import basic_clean, clean_text

log = logging.getLogger("newssort")


class ModelService:
    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self.vectorizer = None
        self.sk_models: dict[str, object] = {}
        self.bert_model = None
        self.bert_tokenizer = None
        self.torch = None
        self.device = "cpu"
        self.status: dict[str, dict] = {
            k: {"available": False, "reason": "not loaded"} for k in config.DISPLAY_NAMES}

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        self._load_sklearn()
        self._load_bert()

    def _load_sklearn(self) -> None:
        vec_path = self.models_dir / config.MODEL_FILES["vectorizer"]
        if vec_path.exists():
            self.vectorizer = joblib.load(vec_path)
        for key in ("logreg", "rf"):
            path = self.models_dir / config.MODEL_FILES[key]
            if self.vectorizer is None:
                self.status[key] = {"available": False,
                                    "reason": f"{config.MODEL_FILES['vectorizer']} not found"}
            elif not path.exists():
                self.status[key] = {"available": False, "reason": f"{path.name} not found"}
            else:
                self.sk_models[key] = joblib.load(path)
                self.status[key] = {"available": True, "reason": ""}

    def _load_bert(self) -> None:
        bert_dir = self.models_dir / config.MODEL_FILES["bert_dir"]
        if not bert_dir.is_dir():
            self.status["bert"] = {"available": False, "reason": f"folder {bert_dir.name}/ not found"}
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            self.status["bert"] = {"available": False,
                                   "reason": "torch / transformers not installed (see requirements-bert.txt)"}
            return
        try:
            self.torch = torch
            self.bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.bert_model.to(self.device).eval()
            self.status["bert"] = {"available": True, "reason": "", "device": self.device}
        except Exception as exc:  # noqa: BLE001 - report any load problem to the UI
            log.exception("Could not load BERT")
            self.status["bert"] = {"available": False, "reason": f"failed to load: {exc}"}

    # ------------------------------------------------------------------ queries
    def available(self) -> list[str]:
        return [k for k in ("bert", "logreg", "rf") if self.status[k]["available"]]

    def predict_proba(self, name: str, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, 4) array of probabilities in config.LABELS order."""
        if not self.status.get(name, {}).get("available"):
            raise KeyError(name)
        if name == "bert":
            return self._bert_proba(texts)
        feats = self.vectorizer.transform([clean_text(t) for t in texts])
        model = self.sk_models[name]
        raw = model.predict_proba(feats)
        out = np.zeros((raw.shape[0], len(config.LABELS)))
        for col, cls in enumerate(model.classes_):       # robust to any class ordering
            out[:, int(cls)] = raw[:, col]
        return out

    def _bert_proba(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        cleaned = [basic_clean(t) for t in texts]
        chunks = []
        for i in range(0, len(cleaned), config.BERT_BATCH_SIZE):
            batch = cleaned[i:i + config.BERT_BATCH_SIZE]
            enc = self.bert_tokenizer(batch, truncation=True, max_length=config.BERT_MAX_LEN,
                                      padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.bert_model(**enc).logits
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(chunks)
