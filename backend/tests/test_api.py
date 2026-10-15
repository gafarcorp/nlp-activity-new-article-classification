import io

from backend.app.main import build_result, csv_safe
import numpy as np


def post(client, **body):
    return client.post("/api/predict", json=body)


def test_health_lists_models(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["models"]["logreg"]["available"] and data["models"]["rf"]["available"]
    assert data["models"]["bert"]["available"] is False   # no BERT folder in the test models
    assert "bert_ag_news" in data["models"]["bert"]["reason"]


def test_predict_single_model(client):
    r = post(client, title="Team wins the championship", description="The coach praised the player after the match.",
             model="logreg")
    assert r.status_code == 200
    body = r.json()
    assert body["primary"]["topic"] == "Sports"
    assert abs(sum(body["primary"]["probabilities"].values()) - 1) < 1e-3
    assert body["agreement"] is None and body["unavailable"] == {}


def test_predict_all_returns_available_models_and_reports_missing_bert(client):
    r = post(client, title="Shares rise as earnings beat investors", model="all")
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"]) == {"logreg", "rf"}
    assert "bert" in body["unavailable"]
    assert body["agreement"] in (True, False)
    assert body["primary"]["model"] == "logreg"


def test_unavailable_model_gives_503_with_reason(client):
    r = post(client, title="anything", model="bert")
    assert r.status_code == 503
    assert "not available" in r.json()["detail"]


def test_empty_input_rejected(client):
    r = post(client, title="  ", description="")
    assert r.status_code == 422
    assert "title or a description" in r.text


def test_too_long_input_rejected(client):
    r = post(client, description="word " * 600)
    assert r.status_code == 422
    assert "too long" in r.text


def test_unknown_model_rejected(client):
    assert post(client, title="x", model="gpt").status_code == 422


def test_no_models_gives_503(empty_client):
    assert empty_client.get("/api/health").json()["status"] == "no_models"
    assert empty_client.post("/api/predict", json={"title": "x", "model": "all"}).status_code == 503


def test_metrics_and_examples_endpoints(client):
    m = client.get("/api/metrics").json()
    assert m["models"]["bert"]["accuracy"] == 94.37
    for k, v in m["models"].items():                     # each confusion matrix sums to 7,600
        assert sum(map(sum, v["confusion"])) == 7600, k
        assert v["errors"] == 7600 - sum(v["confusion"][i][i] for i in range(4)), k
    ex = client.get("/api/examples").json()
    assert len(ex) == 5 and all(e["title"] for e in ex)


def csv_file(text):
    return {"file": ("articles.csv", io.BytesIO(text.encode()), "text/csv")}


def test_batch_json(client):
    text = "Title,Description\nTeam wins match,The coach and player celebrate the goal\n,\nShares rise,Investors like earnings\n"
    r = client.post("/api/predict-batch?model=logreg", files=csv_file(text))
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 3
    assert rows[0]["topic"] == "Sports" and rows[2]["topic"] == "Business"
    assert rows[1]["error"] and rows[1]["topic"] == ""


def test_batch_csv_download_and_formula_protection(client):
    text = 'title,description\n"=HYPERLINK(""http://x"")",earnings shares investors market\n'
    r = client.post("/api/predict-batch?model=rf&format=csv", files=csv_file(text))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "'=HYPERLINK" in r.text and "predicted_topic" in r.text


def test_batch_validation_errors(client):
    assert client.post("/api/predict-batch?model=logreg", files=csv_file("a,b\n1,2\n")).status_code == 422
    assert client.post("/api/predict-batch?model=logreg", files=csv_file("title,description\n")).status_code == 422
    assert client.post("/api/predict-batch?model=bert", files=csv_file("title\nx\n")).status_code == 503
    many = "title\n" + "word\n" * 1001
    assert client.post("/api/predict-batch?model=logreg", files=csv_file(many)).status_code == 422


def test_hints_logic():
    close = build_result("logreg", np.array([0.05, 0.02, 0.48, 0.45]))
    assert "close_call" in close.hints and "business_scitech_overlap" in close.hints
    sure = build_result("bert", np.array([0.97, 0.01, 0.01, 0.01]))
    assert sure.hints == []
    overlap_only = build_result("bert", np.array([0.01, 0.01, 0.30, 0.68]))
    assert overlap_only.hints == ["business_scitech_overlap"]


def test_csv_safe():
    assert csv_safe("=1+1") == "'=1+1" and csv_safe("@x") == "'@x" and csv_safe("hello") == "hello"


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "NewsSort" in r.text
