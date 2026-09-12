# Customer Churn Prediction — Full-Stack ML App

An end-to-end machine learning project: a trained classification model served through a
FastAPI backend, with a web UI where a user enters customer account details and gets a
live churn-risk prediction.

**Live demo:** _add your deployed URL here once live_

## What it does

Predicts whether a telecom customer is likely to cancel their service, using account
signals like contract type, tenure, monthly charges, tech support usage, and recent
support-ticket volume. Returns a churn probability and risk level (Low / Medium / High).

## Tech stack

- **ML:** Python, pandas, NumPy, scikit-learn (Random Forest inside a `Pipeline` with
  `ColumnTransformer` for preprocessing — one artifact handles both feature encoding and
  prediction, so there's no train/serve skew)
- **Backend:** FastAPI, Pydantic for request validation, served with Uvicorn
- **Frontend:** vanilla HTML/CSS/JS calling the `/predict` endpoint
- **CI/CD:** GitHub Actions — installs deps, regenerates data, retrains the model, and
  smoke-tests the API on every push
- **Deployment:** Render (free tier) — see below

## Project structure

churn-predictor/
├── generate_data.py # builds a synthetic but realistic customer dataset
├── train_model.py # trains + evaluates the RandomForest pipeline, saves model.pkl
├── app.py # FastAPI app: /predict, /health, and serves the UI at /app
├── static/index.html # demo frontend
├── requirements.txt
├── render.yaml # Render deployment config
└── .github/workflows/ci.yml

## Model performance

On a held-out 20% test split (synthetic data):

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.77  |
| Precision | 0.69  |
| Recall    | 0.78  |
| F1        | 0.73  |
| ROC-AUC   | 0.86  |

## Run locally

```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
uvicorn app:app --reload
```

