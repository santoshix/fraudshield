# FraudShield — Fraud Detection & Risk Scoring

FraudShield combines a **historical credit-card fraud benchmark** with a separate, interactive **synthetic transaction simulator**. It uses XGBoost, FastAPI, SQLite/PostgreSQL, Kafka, Docker, and a browser dashboard.

![FraudShield dashboard preview](dashboard_mockup.png)

## What it does

- **Historical benchmark (default dashboard tab):** shows results from an XGBoost classifier trained on the anonymized ULB credit-card dataset. With the local CSV and model installed, it also scores sampled held-out transactions.
- **Synthetic demo:** scores simulated transactions using a different eight-feature XGBoost model; shows risk charts, transaction history, and a scoring form.
- **API and persistence:** FastAPI endpoints, SQLite for local use, or PostgreSQL with Docker.
- **Streaming:** Kafka forwards demo transactions to a worker that scores and stores them. Replayed events with the same event ID do not create duplicate database records.
- **Verification:** Python tests, dashboard JavaScript checks, and a Docker integration test run in GitHub Actions.

**Scope:** This is an educational portfolio application, not a production payment processor. The ULB benchmark uses historical anonymized data and has not been validated against current transaction patterns. Synthetic demo scores should not be treated as real fraud decisions.

## Historical model results

Dataset: [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · **283,726 cleaned records** · **56,746 held-out test records**

| Held-out metric | Value |
|---|---:|
| ROC-AUC | **0.9815** |
| Average precision | **0.7906** |
| Precision | **88.33%** |
| Recall | **71.62%** |
| F1 | **0.7910** |
| Brier score | **0.00078** |
| Validation-selected decision threshold | **0.95** |

Data is sorted by transaction time and split chronologically: 60% training, 20% validation, 20% testing. The training portion includes three walk-forward validation folds; the decision threshold is selected on validation data, not the held-out test set. Brier score is reported for evaluation; the probabilities are **not claimed to be calibrated**.

See [the benchmark JSON](benchmarks/real_metrics.json) for the full report and fold results.

## Run locally

Requires Python 3.11 or 3.12. Clone the repository, then run the following from the project folder.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

Open the [dashboard](http://127.0.0.1:8000) or [API docs](http://127.0.0.1:8000/docs). The first launch trains the synthetic model and creates a local SQLite database. No dataset download is needed for the synthetic demo or the historical **summary** cards.

The dashboard opens on **Historical real data**. Choose **Synthetic demo** to try **Analyze transaction**, **Generate 55 demo transactions**, or **Stream event** (direct processing unless Kafka is configured).

For a non-interactive layout preview, open [dashboard_preview.html](dashboard_preview.html). Its controls do not call the model.

### Reproduce historical scoring

The ULB CSV and trained binary files are intentionally not committed to the repository.

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`.
2. Train the historical model:

   ```bash
   python -m ml.train_real
   ```

3. Start `run.py`; open **Historical real data** and select **Load real held-out predictions**.

Training saves `models/fraud_real_xgboost.joblib` and `models/real_metrics.json`. The historical-samples button requires **both** the CSV and these local artifacts. If the artifacts are missing, the UI explains what to install rather than inventing results. Only load model binaries from sources you trust.

Historical API routes:

| Endpoint | Use |
|---|---|
| `GET /api/historical-benchmark` | Published held-out metrics; no local model required |
| `GET /api/historical-samples?limit=20` | Score locally stored held-out rows |
| `GET /api/real-model` | Local historical model metadata |
| `POST /api/score-real` | Score 30 numerical ULB features (`Time`, `V1`–`V28`, `Amount`) |

The synthetic API uses **eight different features** at `POST /api/score`. These models and results are not interchangeable.

### Docker, Kafka, and PostgreSQL

Start Docker Desktop, then run:

```bash
docker compose up --build
```

The stack runs `api`, `postgres`, `kafka`, `worker`, and `streamlit`. The browser dashboard is at http://localhost:8000; optional Streamlit analytics is at http://localhost:8501.

```text
Synthetic demo: Browser → FastAPI → XGBoost → SQLite
Docker ingest:  Browser → FastAPI → Kafka → worker → XGBoost → PostgreSQL
Historical:     ULB features → FastAPI /api/score-real → separate XGBoost model
```

Use `docker compose down` to stop. **`docker compose down -v` also removes stored demo data and model volumes.** Duplicate Kafka deliveries with the same event ID are deduplicated; separate client retries with new IDs are not.

To run Streamlit without Docker, keep the local API running and use:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Tests and safeguards

```bash
python -m pytest -q
```

[GitHub Actions](../../actions) runs Python tests, a JavaScript syntax check, and Docker/Kafka/PostgreSQL integration. These checks validate the implemented paths, not production reliability or real-world fraud detection.

An optional `FRAUDSHIELD_API_KEY` requires `X-API-Key` on guarded API routes. `FRAUDSHIELD_RATE_LIMIT` defaults to 120 requests per minute per IP **per process**. These are local-demo safeguards, not a public authentication or distributed rate-limiting solution. The sample Compose credentials are for development; do not expose the stack publicly without proper security configuration.

## Repository guide

| Path | Purpose |
|---|---|
| `app/main.py`, `app/schemas.py` | FastAPI routes and request validation |
| `app/risk.py`, `app/real_risk.py` | Separate synthetic and historical inference |
| `app/db.py`, `app/worker.py` | Persistence and Kafka consumer |
| `app/static/`, `streamlit_app.py` | Browser dashboard and optional analytics |
| `ml/train.py`, `ml/train_real.py` | Synthetic and historical training pipelines |
| `benchmarks/real_metrics.json` | Published held-out benchmark |
| `tests/`, `compose.yaml` | Automated tests and multi-service setup |

## Project summary

Built an educational fraud-risk scoring application with a FastAPI inference service, XGBoost, Kafka, PostgreSQL, Docker, and two distinct dashboard modes. Evaluated the historical ULB model on 56,746 chronologically held-out transactions (ROC-AUC 0.9815; average precision 0.7906), implemented Kafka event deduplication, and added CI-based tests.

## License

No repository-wide open-source license has been selected. Public visibility does not grant permission to reuse the source.
