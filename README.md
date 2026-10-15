# NewsSort: topic tagging for newsrooms

NewsSort is a web app built on the Activity 2 NLP project (team Neural Nexus). It suggests a topic for a news
article (**World, Sports, Business, Sci/Tech**) with a confidence value, can run **Logistic Regression**,
**Random Forest** and **BERT** side by side, and flags articles a human should check.

It assists an editor; it does not replace one.

## What you get

| Page | What it does |
| --- | --- |
| Classify | Title + description in, topic + confidence + all four probabilities out. "Compare all" shows the three models side by side and warns when they disagree. |
| Batch | Upload a CSV (`title` and/or `description` column), get a sortable, filterable table and a CSV download. |
| Model performance | The real test results from the notebook: table, chart, confusion matrix per model. |
| History | Last 50 predictions, stored only in your browser. "Wrong topic?" records a correction; export them as CSV. |
| About | How it works and its limits. |

Review flags: an article is flagged when the top two probabilities are within 0.15 (`close_call`), or when Business and
Sci/Tech are the top two and the runner-up has at least 0.10 (`business_scitech_overlap`). Thresholds are in
`backend/app/config.py`.

## Quick start (local)

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional, only if you want to serve BERT (install the CPU build of torch from pytorch.org first if you have no GPU):
pip install -r requirements-bert.txt

# copy the files saved by the notebook into backend/models/  (see backend/models/README.txt)
uvicorn backend.app.main:app --reload
```

Open http://127.0.0.1:8000. The green chip in the header shows how many models loaded; hover it for the reason a model
is missing. The first start downloads NLTK data (stopwords, wordnet) if it is not already installed.

### Model files (from the notebook)

| File in `backend/models/` | Written by |
| --- | --- |
| `tfidf_vectorizer.joblib`, `best_model_logreg.joblib`, `random_forest.joblib` | notebook, "Saving the models" cell |
| `bert_ag_news/` | `trainer.save_model("bert_ag_news")` and `tokenizer.save_pretrained("bert_ag_news")` |

The cleaning code in `backend/app/preprocessing.py` is copied from the notebook and **must stay identical to it**,
otherwise the models see different text than they were trained on.

## Docker

```bash
docker compose up --build            # models are mounted read-only from ./backend/models
```

Set `WITH_BERT: "0"` in `docker-compose.yml` to build a smaller image that serves only the two TF-IDF models.

## API

Interactive docs: http://127.0.0.1:8000/docs

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Which models are loaded (and why not, if missing) |
| `POST /api/predict` | `{title, description, model: "bert"\|"logreg"\|"rf"\|"all"}` returns the primary result, per-model results, `agreement`, and `unavailable` |
| `POST /api/predict-batch?model=...&format=json\|csv` | CSV upload (max 1000 rows, 5 MB); one model per call |
| `GET /api/metrics` | The static test results shown on the performance page |
| `GET /api/examples` | Sample articles for the "Try an example" button |

Errors: `422` for invalid input (empty, over 2000 characters, bad CSV), `503` when the requested model is not loaded,
`413` for oversized uploads.

## Configuration (environment variables)

| Variable | Default |
| --- | --- |
| `NEWSSORT_MODELS_DIR` | `backend/models` |
| `NEWSSORT_FRONTEND_DIR` | `frontend` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests train tiny stand-in models, so they check the app's behaviour (validation, hints, batch, errors, CSV
safety), not model accuracy. The real accuracy figures come from the notebook.

## Project layout

```
backend/app/       FastAPI app: main.py, model_service.py, preprocessing.py, schemas.py, config.py, data/
backend/models/    put the trained models here
backend/tests/     pytest suite
frontend/          index.html, styles.css, app.js (no build step, no external scripts)
Dockerfile, docker-compose.yml, requirements*.txt
```

## Security and limits

Done: input length limits; upload size and row limits; all user text rendered with `textContent` (no HTML injection);
spreadsheet-formula protection on CSV downloads; no external scripts or fonts.

Not done (this is a course project, not a production service): authentication, rate limiting, HTTPS, request logging,
multi-user storage (history lives in each browser). BERT inference runs inside the API process, so a very large BERT batch
on a CPU will be slow and block other requests. With several uvicorn workers, each worker loads its own copy of every model.

## Honest notes

- The performance page shows the team's own results: one train/test split, one random seed, and BERT trained on a GPU while
  the other two ran on a CPU, so the times are indicative only.
- The batch page shows an indeterminate progress bar (the whole file is processed in one request).
- About 1 in 18 articles is still misclassified by BERT on the AG News test set, and some labels in that data are ambiguous.

## Credits

Team Neural Nexus, Department of MCA, JAIN (Deemed-to-be University). Dataset: AG News (A. Gulli). Models: scikit-learn,
Hugging Face `bert-base-uncased`.
