@echo off
REM Start MLflow server in a new command prompt window
start "MLflow Server" mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0

prefect server start
prefect cloud login -k pnu_Q9k0SlEhZJQ7sbqHl8GVQ4RrU1g4Z1149ROZ