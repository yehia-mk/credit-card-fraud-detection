# System Architecture

```mermaid
flowchart LR
    A[Credit-card transaction data] --> B[fraud_detection.ipynb\nExploration and analysis]
    A --> C[ML.ipynb\nFeature engineering and training]
    C --> D[Model selection and\nvalidation threshold]
    D --> E[(Saved artifacts\nmodel, scaler, metadata)]

    E --> F[Streamlit app\nfraud_deployment/app.py]
    G[Single transaction\nor batch CSV] --> F
    F --> H[Feature preparation\nV1-V28 + Log_Amount]
    H --> I[Fraud probability\nand threshold decision]
    I --> J[Prediction result\nNormal or Fraud Alert]
    I --> K[(Prediction log\nmetadata only)]
    K --> L[Monitoring tab]

    F --> M[Docker image\nfraud_deployment/Dockerfile]
    M --> N[Container port 8501]
    N --> O[Browser\nlocalhost:8501]
```

## Component Responsibilities

| Component | Responsibility |
| --- | --- |
| `fraud_detection.ipynb` | Explore the transaction data and investigate fraud patterns. |
| `ML.ipynb` | Train candidate models, evaluate them, and select the deployment model and threshold. |
| Saved artifacts | Preserve the model, `Log_Amount` scaler, feature order, model version, and threshold used by the app. |
| `fraud_deployment/app.py` | Validate inputs, prepare features, run inference, display results, and record prediction metadata. |
| Prediction log | Store timestamp, model version, probability, predicted class, and an input hash without raw transaction values. |
| Docker image | Package the Streamlit runtime and deployment files for repeatable execution. |

## Prediction Flow

1. A user enters one transaction or uploads a CSV.
2. The app validates `Amount` and `V1` through `V28`.
3. `Log_Amount = log1p(Amount)` is calculated and transformed with the saved scaler.
4. Features are reordered to match the saved metadata.
5. The model produces a fraud probability.
6. The saved validation threshold converts the probability into `Normal` or `Fraud Alert`.
7. Prediction metadata is appended to the local monitoring log.