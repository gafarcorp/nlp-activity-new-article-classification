"""NewsSort API: classifies a news article with Logistic Regression, Random Forest and/or BERT."""
from __future__ import annotations

import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .model_service import ModelService
from .schemas import ModelResult, PredictRequest, PredictResponse

log = logging.getLogger("newssort")
ORDER = ["bert", "logreg", "rf"]


# ----------------------------------------------------------------------------- helpers
def build_result(name: str, probs: np.ndarray) -> ModelResult:
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    hints: list[str] = []
    if probs[top] - probs[second] < config.CLOSE_CALL_GAP:
        hints.append("close_call")
    pair = {config.LABELS[top], config.LABELS[second]}
    if pair == {"Business", "Sci/Tech"} and probs[second] >= config.OVERLAP_MIN_SECOND:
        hints.append("business_scitech_overlap")
    return ModelResult(
        model=name, model_name=config.DISPLAY_NAMES[name],
        topic=config.LABELS[top], confidence=round(float(probs[top]), 4),
        probabilities={lab: round(float(p), 4) for lab, p in zip(config.LABELS, probs)},
        hints=hints)


def csv_safe(value: str) -> str:
    """Stop spreadsheet programs from running text such as '=1+1' as a formula."""
    value = "" if value is None else str(value)
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def create_app(models_dir: Path | str | None = None) -> FastAPI:
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = ModelService(models_dir)
        service.load()
        app.state.service = service
        log.info("Models available: %s", service.available())
        yield

    app = FastAPI(title="NewsSort", version="1.0.0", lifespan=lifespan,
                  description="Topic tagging for news articles (World, Sports, Business, Sci/Tech).")
    api = APIRouter(prefix="/api")

    def svc() -> ModelService:
        return app.state.service

    # ------------------------------------------------------------------ endpoints
    @api.get("/health")
    def health():
        s = svc()
        return {"status": "ok" if s.available() else "no_models",
                "models": {k: {"name": config.DISPLAY_NAMES[k], **s.status[k]} for k in ORDER},
                "limits": {"max_chars": config.MAX_CHARS, "max_batch_rows": config.MAX_BATCH_ROWS}}

    @api.get("/metrics")
    def metrics():
        return json.loads((config.DATA_DIR / "metrics.json").read_text(encoding="utf-8"))

    @api.get("/examples")
    def examples():
        return json.loads((config.DATA_DIR / "examples.json").read_text(encoding="utf-8"))

    @api.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        s = svc()
        wanted = ORDER if req.model == "all" else [req.model]
        results: dict[str, ModelResult] = {}
        unavailable: dict[str, str] = {}
        for name in wanted:
            if name in s.available():
                results[name] = build_result(name, s.predict_proba(name, [req.text])[0])
            else:
                unavailable[name] = s.status[name]["reason"]
        if not results:
            reasons = "; ".join(f"{config.DISPLAY_NAMES[k]}: {v}" for k, v in unavailable.items())
            raise HTTPException(503, f"The requested model is not available. {reasons}")
        primary = results[next(k for k in ORDER if k in results)]
        agreement = None if len(results) < 2 else len({r.topic for r in results.values()}) == 1
        return PredictResponse(primary=primary, results=results, agreement=agreement,
                               unavailable=unavailable)

    @api.post("/predict-batch")
    async def predict_batch(file: UploadFile = File(...),
                            model: Literal["bert", "logreg", "rf"] = Query("logreg"),
                            format: Literal["json", "csv"] = Query("json")):
        s = svc()
        if model not in s.available():
            raise HTTPException(503, f"Model '{model}' is not available: {s.status[model]['reason']}")
        raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
        if len(raw) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(422, "The file must be UTF-8 encoded text (CSV).")
        reader = csv.DictReader(io.StringIO(text))
        fields = {(f or "").strip().lower(): f for f in (reader.fieldnames or [])}
        if "title" not in fields and "description" not in fields:
            raise HTTPException(422, "The CSV needs a 'title' and/or a 'description' column.")
        rows = []
        for rec in reader:
            rows.append({"title": (rec.get(fields.get("title", ""), "") or "").strip(),
                         "description": (rec.get(fields.get("description", ""), "") or "").strip()})
            if len(rows) > config.MAX_BATCH_ROWS:
                raise HTTPException(422, f"Too many rows. The limit is {config.MAX_BATCH_ROWS}.")
        if not rows:
            raise HTTPException(422, "The CSV has no data rows.")

        usable = [i for i, r in enumerate(rows)
                  if 0 < len(r["title"]) + len(r["description"]) <= config.MAX_CHARS]
        out = [{"row": i + 1, **r, "topic": "", "confidence": None, "review": False,
                "error": "empty or too long" if i not in usable else ""} for i, r in enumerate(rows)]
        if usable:
            probs = s.predict_proba(model, [f"{rows[i]['title']} {rows[i]['description']}".strip() for i in usable])
            for i, p in zip(usable, probs):
                res = build_result(model, p)
                out[i].update(topic=res.topic, confidence=res.confidence, review=bool(res.hints))

        if format == "csv":
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["row", "title", "description", "predicted_topic", "confidence", "review_flag", "model", "error"])
            for r in out:
                w.writerow([r["row"], csv_safe(r["title"]), csv_safe(r["description"]), r["topic"],
                            "" if r["confidence"] is None else r["confidence"],
                            "yes" if r["review"] else "", model, r["error"]])
            return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                     headers={"Content-Disposition": 'attachment; filename="newssort_predictions.csv"'})
        return {"model": model, "model_name": config.DISPLAY_NAMES[model], "count": len(out), "rows": out}

    app.include_router(api)
    if config.FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
