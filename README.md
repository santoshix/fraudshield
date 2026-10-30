# FraudShield — Real-Time Fraud Detection & Risk Engine

A complete **student portfolio project** with XGBoost machine learning, FastAPI REST API, a visual browser dashboard, SQL storage, a Streamlit analytics dashboard, and optional Kafka streaming + PostgreSQL with Docker.

> **IMPORTANT:** Training data is entirely **synthetic** (invented examples with simulated labels). Risk scores are model estimates **not verified fraud**. This application is for learning only. **Never use for actual payments, fraud investigations or financial decisions.** It does not implement real payment processing or enterprise-grade security/monitoring.

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

## 6. What's actually happening?

`Transaction inputs → 8 numeric features → XGBoost probability → risk score 0–100 → risk status → database → dashboards`

- **LOW RISK:** risk score less than 35
- **REVIEW:** 35 to less than 70
- **HIGH RISK:** 70 or higher

These are **illustrative, not calibrated banking thresholds**. The AI predicts risk from the fake pattern rules used to make fake labeled training data. It cannot be honestly described as detecting real fraud accurately.

**Kafka version:** Dashboard posts to FastAPI `/api/ingest` → Kafka `fraud.incoming` topic → Kafka consumer → model inference → PostgreSQL → dashboard refresh.

## 7. Resume / interview explanation

> "Built an end-to-end educational fraud-risk scoring platform using Python, XGBoost, FastAPI, SQLAlchemy and PostgreSQL. Implemented real-time streaming ingestion through Kafka, a background consumer, risk classifications, and monitoring dashboards. Used entirely synthetic training data and reported validation metrics with clear limitations."

**To turn this into a stronger portfolio:** add a suitably licensed, real-world dataset with an appropriate label definition, time-based validation, severe class imbalance handling, drift monitoring, calibration checks, tests for duplicate Kafka deliveries and idempotent writes, authentication, secret management, and responsible decision-review policies. Deploy only with properly configured cloud security and budgets.

## 8. Troubleshooting

- `ModuleNotFoundError`: install requirements using the project's `.venv` Python, not a different interpreter.
- Port 8000 busy: stop the other application using port 8000.
- Model takes some time to train on first launch: the terminal will print the training result, then the server will start.
- Streamlit says cannot connect: start `python run.py` first.
- Kafka unavailable: run `docker compose up --build` for the full pipeline or use the ordinary **Analyze transaction** button.
- Docker build may download large images and packages; this is normal for the full stack.
