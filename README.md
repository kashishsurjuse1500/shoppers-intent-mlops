# Shoppers Intent MLOps

A production-grade end-to-end ML Classification project that predicts whether an online shopper will make a purchase or not.

## Live Demo
- **Frontend**: https://shoppers-intent-mlops.onrender.com
- **Swagger UI**: https://shoppers-intent-mlops.onrender.com/docs

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Source DB | MySQL |
| Data Warehouse | PostgreSQL |
| ETL | Python (Extract → Transform → Load) |
| Data Validation | Pandas |
| ML Models | Random Forest, XGBoost, LightGBM |
| Hyperparameter Tuning | GridSearchCV |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API | FastAPI |
| Monitoring | Evidently AI |
| Scheduler | APScheduler |
| Logging | Python logging module |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Registry | Docker Hub |
| Deployment | Render Free Tier |

## Architecture


## Project Structure

shoppers-intent-mlops/
├── api/
│   ├── templates/index.html     # frontend html
│   ├── static/style.css         # frontend css
│   ├── static/script.js         # frontend javascript
│   └── main.py                  # fastapi app
├── data/
│   ├── online_shoppers_intention.csv  # raw dataset
│   └── reload_data.py           # csv to mysql loader
├── etl/
│   ├── extract.py               # extract from mysql
│   ├── transform.py             # clean, validate, encode
│   └── load.py                  # load to postgresql
├── ml/
│   ├── preprocess.py            # scaling, selection, smote
│   ├── train.py                 # train rf+xgb+lgbm with tuning
│   ├── evaluate.py              # evaluate saved model
│   ├── predict.py               # make predictions
│   └── mlflow_tracker.py        # mlflow tracking + registry
├── monitoring/
│   └── drift_report.py          # evidently drift report
├── scheduler/
│   └── retrain_job.py           # apscheduler auto retrain
├── tests/
│   └── test_api.py              # pytest tests
├── utils/
│   └── logger.py                # common logging module
├── logs/                        # log files
├── .github/workflows/
│   └── ci_cd.yml                # github actions ci/cd
├── .env.example                 # env variables template
├── .gitignore                   # git ignore rules
├── .dockerignore                # docker ignore rules
├── Dockerfile                   # docker configuration
├── render.yaml                  # render deployment config
└── requirements.txt             # python dependencies




## Dataset
- **Source**: Online Shoppers Purchasing Intention Dataset
- **Rows**: 12,330
- **Target**: Revenue (Will Purchase or Not)
- **Features**: 17 behavioral and session features
- **Class Imbalance**: Fixed using SMOTE

## ML Pipeline
1. **Extract** — Raw data from MySQL
2. **Validate** — Schema, nulls, value ranges checked
3. **Transform** — Encode, clean, deduplicate
4. **Load** — Push to PostgreSQL warehouse
5. **Preprocess** — StandardScaler + SelectKBest + SMOTE
6. **Train** — RF + XGBoost + LightGBM with GridSearchCV
7. **Select** — Best model by ROC AUC auto-selected
8. **Track** — MLflow experiment tracking + model registry
9. **Predict** — FastAPI prediction endpoint
10. **Monitor** — Evidently data drift report
11. **Retrain** — APScheduler daily 2 AM auto retrain

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 91.82% |
| Precision | 89.96% |
| Recall | 94.33% |
| F1 Score | 92.09% |
| ROC AUC | 97.32% |

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Frontend UI |
| GET | /health | Health check |
| POST | /predict | Make prediction |
| GET | /docs | Swagger UI |

## Setup and Run Locally

### Prerequisites
- Python 3.11
- MySQL 8.0
- PostgreSQL 17+
- Docker (optional)

### Steps
```bash
# clone repository
git clone https://github.com/kashishsurjuse/shoppers-intent-mlops.git
cd shoppers-intent-mlops

# create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# install dependencies
pip install -r requirements.txt

# setup environment variables
cp .env.example .env
# edit .env with your credentials

# load data into MySQL
python data/reload_data.py

# run ETL pipeline (MySQL → PostgreSQL)
python etl/load.py

# train model
python ml/train.py

# run API
uvicorn api.main:app --reload
```

### Run with Docker
```bash
docker build -t shoppers-intent-mlops .
docker run -p 8000:8000 --env-file .env shoppers-intent-mlops
```

### Run Tests
```bash
pytest tests/test_api.py -v
```

### Run Scheduler
```bash
python -c "from scheduler.retrain_job import run_pipeline; run_pipeline()"
```

### Generate Drift Report
```bash
python monitoring/drift_report.py
```

### View MLflow UI
```bash
mlflow ui
# open http://127.0.0.1:5000
```

## CI/CD Pipeline
1. Push to main → GitHub Actions triggered
2. MySQL + PostgreSQL services spin up
3. Data loaded → ETL run → Model trained
4. Pytest tests run
5. Docker image built and pushed to Docker Hub
6. Render auto deploys latest image

## Environment Variables


## Logging
All modules log to:
- **Console** — real time output
- **`logs/app.log`** — persistent log file

## Author
kashish surjuse