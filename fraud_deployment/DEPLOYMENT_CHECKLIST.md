# Deployment Checklist

## A. Before deployment
- [ ] Run the ML notebook top-to-bottom.
- [ ] Confirm `models/best_ml_model_latest.joblib` exists.
- [ ] Confirm `models/amount_scaler.joblib` exists.
- [ ] Confirm `models/model_metadata.joblib` exists.
- [ ] Confirm metadata contains the expected feature columns and threshold.
- [ ] Put the deployment files in the project repository root.

## B. Local validation
- [ ] Install `requirements.txt`.
- [ ] Run `python -m pytest tests -q`.
- [ ] Run `streamlit run app.py`.
- [ ] Test one transaction.
- [ ] Download the CSV template.
- [ ] Test batch prediction.
- [ ] Confirm `reports/prediction_log.csv` is created.
- [ ] Open Monitoring and confirm logged predictions appear.

## C. Docker
- [ ] Run `docker build -t credit-card-fraud .`.
- [ ] Run `docker run --rm -p 8501:8501 credit-card-fraud`.
- [ ] Open the app in the browser.
- [ ] Test one prediction again.

## D. GitHub
- [ ] Create the repository.
- [ ] Add files.
- [ ] Commit.
- [ ] Push to `main`.
- [ ] Confirm README renders correctly.
- [ ] Confirm no API keys/passwords are present.
- [ ] Confirm the huge raw dataset is not committed.

## E. Final demo
- [ ] Explain model version.
- [ ] Explain threshold.
- [ ] Show single prediction.
- [ ] Show batch prediction.
- [ ] Show logging.
- [ ] Show monitoring.
- [ ] Show Docker setup.
- [ ] Show GitHub repository.
