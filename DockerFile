FROM ghcr.io/mlflow/mlflow:v2.11.3
CMD mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 10000
