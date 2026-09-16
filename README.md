# Credit Card Fraud Detection

An end-to-end credit card fraud detection project with notebook-based model development and a Streamlit deployment.

## Project Overview

The project trains and evaluates machine-learning models on anonymized credit-card transactions, selects a decision threshold on validation data, and serves the saved model through a Streamlit application.

The deployment is designed as a fraud-screening prototype. It should not be treated as a certified banking authorization or production financial decision system.

**Live application:** [Open the Streamlit app](https://credit-card-fraud-detection-m6bnc6drabqmgwugdfk4wd.streamlit.app/)

## Repository Structure

```text
.
├── fraud_detection.ipynb       # Data exploration and fraud-analysis work
├── ML.ipynb                    # Model training, evaluation, and artifact creation
├── models/                     # Saved model, scaler, and metadata artifacts
├── reports/                    # Evaluation reports and deployment monitoring output
├── fraud_deployment/
│   ├── app.py                  # Streamlit application
│   ├── Dockerfile              # Container image definition
│   ├── requirements.txt        # Runtime dependencies
│   ├── requirements-dev.txt    # Test dependencies
│   ├── models/                 # Deployment model artifacts
│   ├── reports/                # Deployment reports and prediction log
│   └── tests/                  # Deployment tests
├── docs/
│   └── architecture.md        # System architecture diagram
└── README.md
```

## Architecture

The system architecture is documented in [docs/architecture.md](docs/architecture.md). In brief:

1. The notebooks prepare data and train candidate models.
2. The selected model, scaler, and metadata are saved as versioned artifacts.
3. The Streamlit app loads those artifacts and applies the saved feature order and threshold.
4. Users can score one transaction or upload a CSV for batch scoring.
5. Prediction metadata is logged without storing raw transaction values.

## Run Locally

From the deployment directory:

```powershell
cd fraud_deployment
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` in a browser.

## Run with Docker

Docker must be available and running. Build from the deployment directory so the Docker build context contains the application, models, and reports:

```powershell
cd fraud_deployment
docker build -t credit-card-fraud .
docker run --rm --name credit-card-fraud-app -p 8501:8501 credit-card-fraud
```

Open `http://localhost:8501`.

## Run Tests

```powershell
cd fraud_deployment
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

The tests check artifact availability, model compatibility, metadata fields, feature preparation, and valid probability output.

## Deploy with Streamlit Community Cloud

The repository is ready to deploy from [Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in with the GitHub account that owns the repository.
2. Select **New app**.
3. Choose repository `yehia-mk/credit-card-fraud-detection`.
4. Select branch `main`.
5. Set **Main file path** to `fraud_deployment/app.py`.
6. Select **Deploy**.

The root `requirements.txt` installs the runtime dependencies used by the nested Streamlit app. The model artifacts required at startup are already committed under `fraud_deployment/models/`.

After deployment, Streamlit will provide a public app URL. Keep the repository private if the model artifacts or project data should not be publicly accessible.

## Model Artifacts

The application expects these files in `fraud_deployment/models/`:

- `best_ml_model_latest.joblib`
- `amount_scaler.joblib`
- `model_metadata.joblib`

The model uses `V1` through `V28` and `Log_Amount`. The app derives `Log_Amount` from the input `Amount`, applies the saved scaler, and reads the validation-selected threshold from the metadata file.

## Git and Data Policy

Raw datasets, generated prediction logs, notebook checkpoints, training logs, virtual environments, and Python cache files are excluded from version control. Model artifacts and static evaluation reports may be committed when they are required to reproduce the deployment.

Before publishing, review `git status` and confirm that no credentials, `.env` files, or large raw datasets are staged.

## License

Add the project team's chosen license before public distribution.