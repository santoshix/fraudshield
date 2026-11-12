# FraudShield — Fraud Risk Scoring & Streaming Demo

An educational **student portfolio project** for synthetic-transaction risk scoring using XGBoost, FastAPI, SQLite, and Streamlit. An optional Docker Compose setup adds Kafka, a background consumer, and PostgreSQL; the local quick start scores transactions directly without a queue.

> **IMPORTANT:** The default dashboard's training data is **synthetic** (invented examples with simulated labels). Risk scores are model estimates **not verified fraud**. This application is for learning only. **Never use for actual payments, fraud investigations or financial decisions.** It does not implement real payment processing or enterprise-grade security/monitoring.

**Want to see the design immediately?** Open `dashboard_preview.html` in your browser or view `dashboard_mockup.png`. It includes preloaded sample results and requires no Python, but its buttons do not run the actual AI. For working predictions, follow the setup below.

## 1. Easiest start: just 3 steps

Install **Python 3.11 or 3.12** and **VS Code**. Open this project's folder in VS Code > Terminal.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

If `py -3.11` isn't available, try `python -m venv .venv` and `.venv\Scripts\python.exe` for the next commands.

**Mac / Linux:**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

### Open these links in your browser

- **MAIN DASHBOARD** → http://127.0.0.1:8000
- **API DOCUMENTATION** → http://127.0.0.1:8000/docs

The **first run auto-trains** XGBoost and writes its model to `models/`, creating the SQLite database `fraud.db`. No dataset download, API key, paid subscription, credit card or cloud account is required.

### Things to click

1. Click **Generate 55 demo transactions** to populate all the charts and table.
2. Click **Analyze transaction** with the example inputs to generate a predicted risk score.
3. Change amount, distance, velocity, international/online flags, and predict again.
4. Click **Stream event**. In the quick-start mode it scores directly. In Docker it sends a Kafka message that the separate worker consumes.
5. Use `/docs` to see and test every REST API endpoint.

## 2. Optional: Streamlit interactive analytics dashboard

Keep `run.py` running in the first terminal. In a **second** terminal, from the same folder:

**Windows:** `.venv\Scripts\python.exe -m streamlit run streamlit_app.py`

**Mac / Linux:** `.venv/bin/python -m streamlit run streamlit_app.py`

Open http://127.0.0.1:8501 for interactive charts, transactions, model metrics and a separate test form.

## 3. Optional: FULL system with Docker, PostgreSQL, Kafka and separate worker

Install Docker Desktop, start Docker, and in the project folder run:

```bash
docker compose up --build
```

After the containers start, visit:

- Main browser UI: http://localhost:8000
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

This starts **five** services:

- `api` — FastAPI scoring + browser dashboard
- `postgres` — database for results
- `kafka` — transaction queue in KRaft mode
- `worker` — separate Kafka consumer scoring queued events
- `streamlit` — analytics dashboard

The main dashboard's **Stream event** button now sends transactions to Kafka. The worker scores and stores them asynchronously; press **Refresh data** or wait for automatic refresh. Use `docker compose down` to stop, or `docker compose down -v` to also DELETE saved demo data/models.

Note: Docker setup is comparatively resource-intensive; start with the easy Python + SQLite mode if you're new.

## 4. Project structure

```
fraud-risk-engine/
  README.md                ← you're here
  requirements.txt         ← packages
  run.py                   ← easiest API + browser dashboard launcher
  streamlit_app.py         ← extra analytics UI
  compose.yaml             ← 5 Docker services
  Dockerfile               ← build Python service image
  .env.example             ← optional environment settings
  app/
    main.py                ← REST endpoints, scoring, demo data and Kafka producer
    risk.py                ← load AI model and calculate risk
    schemas.py             ← input validation
    db.py                  ← SQLite or PostgreSQL models
    worker.py              ← Kafka transaction consumer
    static/
      index.html           ← main visual dashboard
      styles.css           ← dashboard design
      app.js               ← dashboard interaction and charts
  ml/
    train.py               ← train XGBoost on synthetic examples
  tests/
    test_api.py            ← basic automated tests
```

## 5. Run tests

From the project folder, with dependencies installed:

**Windows:** `.venv\Scripts\python.exe -m pytest -q`

**Mac / Linux:** `.venv/bin/python -m pytest -q`

GitHub Actions runs the test suite for pushes and pull requests (see `.github/workflows/tests.yml`). The tests cover API responses and input validation. They are not a substitute for end-to-end Kafka/PostgreSQL integration tests, which have not been established here.

## 6. What's actually happening?

`Transaction inputs → 8 numeric features → XGBoost probability → risk score 0–100 → risk status → database → dashboards`

- **LOW RISK:** risk score less than 35
- **REVIEW:** 35 to less than 70
- **HIGH RISK:** 70 or higher

These are **illustrative, not calibrated banking thresholds**. The AI predicts risk from the fake pattern rules used to make fake labeled training data. It cannot be honestly described as detecting real fraud accurately.

**Kafka version:** Dashboard posts to FastAPI `/api/ingest` → Kafka `fraud.incoming` topic → Kafka consumer → model inference → PostgreSQL → dashboard refresh.

## 7. Resume / interview explanation

> "Built an educational fraud-risk scoring API and analytics dashboards with XGBoost, FastAPI, and SQLite, trained and evaluated on synthetic transactions. Implemented a separate Docker Compose streaming example using Kafka, a worker, and PostgreSQL; streaming is not used in the default local mode."

**To turn this into a stronger portfolio:** add a suitably licensed, real-world dataset with an appropriate label definition, time-based validation, severe class imbalance handling, drift monitoring, calibration checks, tests for duplicate Kafka deliveries and idempotent writes, authentication, secret management, and responsible decision-review policies. Deploy only with properly configured cloud security and budgets.

## 8. Troubleshooting

- `ModuleNotFoundError`: install requirements using the project's `.venv` Python, not a different interpreter.
- Port 8000 busy: stop the other application using port 8000.
- Model takes some time to train on first launch: the terminal will print the training result, then the server will start.
- Streamlit says cannot connect: start `python run.py` first.
- Kafka unavailable: run `docker compose up --build` for the full pipeline or use the ordinary **Analyze transaction** button.
- Docker build may download large images and packages; this is normal for the full stack.

## Historical fraud benchmark — measured results

The optional real-transaction XGBoost pipeline has now been **run on the ULB anonymized credit-card fraud dataset**. The published held-out results are saved in [`benchmarks/real_metrics.json`](benchmarks/real_metrics.json). These are outcomes from a chronological **60% training / 20% validation / 20% test** split on **283,726 rows after cleaning** (56,746 held-out test transactions).

| Metric | Held-out result |
|---|---:|
| ROC-AUC | **0.9815** |
| Average precision (PR-AUC) | **0.7906** |
| Precision | **88.33%** |
| Recall | **71.62%** |
| F1 score | **0.7910** |
| Brier score | **0.00078** |
| Selected threshold (validation set) | **0.95** |

Three walk-forward training-period validation folds are also recorded in the JSON. This is a **historical research benchmark**, not evidence of safe production credit-card fraud detection. The base rate is extremely low, and model calibration/drift on future transactions are unverified.

### Use the trained model in the same FraudShield application

The repository intentionally excludes the original CSV and serialized model binary. They must be kept outside Git. If you have the separately supplied `FraudShield_Real_Model_Ready.zip`, extract the archive into the **project root** so it produces:

```text
models/
  fraud_real_xgboost.joblib
  real_metrics.json
```

Then run `python run.py` using the project's virtual environment; visit `http://127.0.0.1:8000/api/real-model` for the real-model metrics or `http://127.0.0.1:8000/docs` for `POST /api/score-real` (requires 30 anonymized feature values). To reproduce training instead, put `creditcard.csv` in `data/` and run `python -m ml.train_real`. **Only load model binaries you trust**, because `joblib` files can execute code when loaded.

## 9. Real historical fraud benchmark (one integrated FraudShield application)

FraudShield now contains an **optional real-data XGBoost mode inside the same FastAPI app**. It uses the public, anonymized ULB historical credit-card dataset (30 columns: Time, V1–V28, Amount), which is not interchangeable with the existing eight-feature synthetic demo. This is one repository and one application, not a second project.

1. Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud, following the dataset's usage conditions.
2. Place it in `data/creditcard.csv` (not committed to Git).
3. From the repository root, run `python -m ml.train_real` with your virtual environment.
4. Restart FraudShield, then open `/api/real-model` to review metrics and `/docs` to try `POST /api/score-real` with all 30 named feature values.

The historical benchmark uses chronological train/validation/test splits; walk-forward validation within the training partition; a review threshold chosen on validation F1; and held-out ROC-AUC, average precision, precision, recall, F1 and Brier score. The ULB dataset is historical and anonymized, so these results do not prove real-time banking accuracy. The dataset must be downloaded locally and the training script run before this mode becomes available. The current eight-feature UI remains an educational demo.

## 10. Security and Kafka reliability

- To require an API key for non-health API endpoints, set `FRAUDSHIELD_API_KEY` in your local environment and send it as `X-API-Key`. **Never put it in browser JavaScript or Git**. The existing browser dashboard does not support authenticated sessions; leaving the key unset is for trusted, local demos only.
- `FRAUDSHIELD_RATE_LIMIT` defaults to 120 requests/minute per IP **per server process**. This in-memory limit is not distributed protection; a production deployment needs a reverse proxy, shared rate limiting, TLS, proper authentication and secret management.
- Kafka messages now contain an event ID and the consumer writes a stable unique ID to the database. Replaying the same event cannot create a second record, and the consumer commits after the write is confirmed. The producer should retain a stable event ID when clients retry uncertain sends for full end-to-end deduplication.
