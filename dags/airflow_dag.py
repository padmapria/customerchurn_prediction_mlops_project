from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from data_processing.load_data import load_data
from data_processing.preprocess import preprocess_data
from model.baseline_model import train_baseline_model
from model.hyperparameter_tuning import optimize_hyperparameters
from config.main_config import get_config

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "mlops_pipeline",
    default_args=default_args,
    description="MLOps Pipeline with Apache Airflow",
    schedule_interval="@daily",  # Adjust the schedule interval as needed
)

# Task to load data
def load_data_task():
    cfg = get_config()
    data_path = cfg.data.path
    start_date = "2022-01-01"
    end_date = "2022-01-31"
    loaded_data = load_data(data_path, start_date, end_date)
    return loaded_data

# Task to preprocess data
def preprocess_data_task(data):
    X_train, X_test, y_train, y_test = preprocess_data(data)
    return X_train, X_test, y_train, y_test

# Task to train the baseline model
def train_baseline_model_task(X_train, y_train):
    baseline_model = train_baseline_model(X_train, y_train)
    return baseline_model

# Task to optimize hyperparameters
def optimize_hyperparameters_task(X_train, y_train):
    best_hyperparameters = optimize_hyperparameters(X_train, y_train)
    return best_hyperparameters

# Define the DAG tasks
load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data_task,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id="preprocess_data_task",
    python_callable=preprocess_data_task,
    provide_context=True,
    dag=dag,
)

train_baseline_model_task = PythonOperator(
    task_id="train_baseline_model_task",
    python_callable=train_baseline_model_task,
    provide_context=True,
    dag=dag,
)

optimize_hyperparameters_task = PythonOperator(
    task_id="optimize_hyperparameters_task",
    python_callable=optimize_hyperparameters_task,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
load_data_task >> preprocess_data_task
preprocess_data_task >> [train_baseline_model_task, optimize_hyperparameters_task]

