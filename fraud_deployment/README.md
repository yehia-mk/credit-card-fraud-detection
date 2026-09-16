# Credit Card Fraud Detection — Deployment

This folder is the deployment layer for the team's Credit Card Fraud Detection project.

## What this part implements

The graduation-project specification requires a working Streamlit interface, versioned model artifacts, reproducible requirements, prediction logging, Docker setup, and a clean GitHub repository.

This deployment provides:

- Streamlit single-transaction prediction UI
- Batch CSV scoring
- Validation-selected fraud threshold loaded from metadata
- Exact feature-order validation
- Saved scaler reuse for `Log_Amount`
- Model/version display
- Prediction logging without storing raw transaction values
- Basic monitoring page
- CSV template download
- Dockerfile
- Basic automated deployment tests
- GitHub-ready README and repository structure

The project specification explicitly requires a working Streamlit interface, versioned model, reproducible requirements, prediction logging, Docker setup, and GitHub publication. See the source specification for the mandatory checklist.

## IMPORTANT: run the ML notebook first

The ML notebook already creates these artifacts:

```text
models/best_ml_model_latest.joblib
models/amount_scaler.joblib
models/model_metadata.joblib
```

It also records the trained feature order and threshold in `model_metadata.joblib`.

A versioned model artifact may also be present:

```text
models/best_ml_model_vYYYYMMDD_HHMMSS.joblib
```

For example:

```text
models/best_ml_model_v20260916_003350.joblib
```

Do NOT retrain or recreate the model inside `app.py`.

## Expected input

The trained model uses:

```text
V1, V2, ..., V28, Log_Amount
```

The app accepts:

```text
V1 ... V28
Amount
Time (optional)
```

`Log_Amount` is created inside the app using:

```text
log1p(Amount)
```

and the already-fitted `RobustScaler` from the ML notebook is applied.

`Time` is intentionally not passed to the model because the ML notebook excludes `Time` from `feature_cols`.

## Folder placement

Put these deployment files in the root of the project repository:

```text
credit-card-fraud/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── models/
│   ├── best_ml_model_latest.joblib
│   ├── best_ml_model_vYYYYMMDD_HHMMSS.joblib
│   ├── amount_scaler.joblib
│   └── model_metadata.joblib
├── reports/
│   └── prediction_log.csv          # generated after predictions
├── tests/
│   └── test_app.py
├── notebooks/
│   └── ...                         # team's notebooks
└── data/
    └── ...                         # keep raw data local; do not upload huge datasets
```

The source project specification recommends a repository containing `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `tests/`, `requirements.txt`, `.env.example`, and `README.md`.

## 1. Local run

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## 2. Run with Docker

Build:

```bash
docker build -t credit-card-fraud .
```

Run:

```bash
docker run --rm -p 8501:8501 credit-card-fraud
```

Open:

```text
http://localhost:8501
```

Docker execution should be verified on a machine with Docker available.

## 3. Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The tests verify that the saved model can be loaded, metadata is structurally valid, required features are present, and the app's feature preparation produces the exact saved feature order.

## 4. Prediction logging and monitoring

Predictions are logged locally in:

```text
reports/prediction_log.csv
```

The log records prediction-related information such as:

- timestamp
- model version
- fraud probability
- predicted class
- input hash

Raw transaction values are not stored in the prediction log.

The Streamlit Monitoring page provides basic monitoring information such as:

- number of logged predictions
- fraud alerts
- alert rate
- average prediction probability

The prediction log is generated during application use and is not required to be committed to GitHub.

## 5. GitHub

Create an empty GitHub repository, then from the project root:

```bash
git init
git add .
git commit -m "Add fraud detection deployment"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the repository URL created by the team.

After pushing, the repository should contain the app, Dockerfile, requirements, tests, README, and the model artifacts required by the deployment.

## 6. Before pushing

Check:

```bash
git status
```

Make sure you did NOT accidentally add:

- API keys
- passwords
- `.env` files
- the huge raw Kaggle dataset
- temporary notebook checkpoints
- virtual environments
- unnecessary training logs
- generated local CSV logs that are not intended for version control

The `.gitignore` file excludes raw datasets, CSV files, secrets, virtual environments, Python cache files, and common local logs.

The source project specification specifically says to keep secrets/API keys outside source control and not to upload huge raw datasets.

## 7. Deployment demo checklist

For the final demo, show:

1. GitHub repository
2. `app.py`
3. `models/model_metadata.joblib`
4. Streamlit single-transaction prediction
5. Batch CSV upload
6. Prediction result and probability
7. Prediction log
8. Monitoring tab
9. Docker build/run
10. Final deployed/local URL

## Important modeling note

The ML notebook selected the final decision threshold using the validation set and then evaluated the test set once. The deployment must use the saved threshold from metadata rather than choosing a new threshold inside the app.

The model should be described as a fraud-screening prototype. It should not be presented as a certified banking authorization or production financial decision system.


## Deployment status

The local deployment has been verified with:

- Streamlit application
- Single-transaction prediction
- Batch CSV scoring
- Prediction logging
- Monitoring page
- Saved model artifacts
- Automated deployment tests

Automated deployment tests currently pass successfully.

Docker build/run should be verified separately on a machine with Docker available.
