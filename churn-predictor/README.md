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

```
churn-predictor/
├── generate_data.py      # builds a synthetic but realistic customer dataset
├── train_model.py        # trains + evaluates the RandomForest pipeline, saves model.pkl
├── app.py                 # FastAPI app: /predict, /health, and serves the UI at /app
├── static/index.html      # demo frontend
├── requirements.txt
├── render.yaml             # Render deployment config
└── .github/workflows/ci.yml
```

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

Open http://localhost:8000/app for the UI, or http://localhost:8000/docs for the
interactive API docs.

## Deploy live (Render, free, ~10 minutes)

1. Push this folder to a new GitHub repo.
2. Go to https://render.com → sign in with GitHub → **New +** → **Web Service**.
3. Connect your repo. Render will detect `render.yaml` automatically and configure:
   - Build command: `pip install -r requirements.txt && python generate_data.py && python train_model.py`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Click **Create Web Service**. First build takes ~3–5 minutes.
5. Once live, your app is at `https://<your-app-name>.onrender.com/app`.
6. Add that URL to this README and to your resume project bullet.

Note: Render's free tier spins down after inactivity and takes ~30s to wake up on the
first request — mention this if a recruiter tries it and it's slow to load initially.

## Possible extensions (good talking points for interviews)

- Swap the synthetic dataset for a real one (e.g. IBM Telco Customer Churn) and compare
  model performance.
- Add SHAP values to explain individual predictions (which features drove this specific
  customer's risk score).
- Add a `/retrain` endpoint or a scheduled GitHub Action to retrain on new data.
- Track experiments with MLflow.
