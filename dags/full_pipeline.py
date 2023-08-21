import mlflow
import prefect
import docker
from prefect import Flow, task
from data_processing.load_data import load_data
from data_processing.training_data_preprocess import preprocess_data
from model.baseline_model import train_baseline_model
from model.hyperparameter_tuning import optimize_hyperparameters

# Define MLflow tracking URI
mlflow_tracking_uri = "your_mlflow_tracking_uri"  # Replace with the actual URI

# MLflow Setup
mlflow.set_tracking_uri(mlflow_tracking_uri)

@task
def build_docker_image():
    client = docker.from_env()
    client.images.build(path="./", tag="my-model-image")

@task
def deploy_docker_image():
    client = docker.from_env()
    client.containers.run("my-model-image", detach=True, ports={"5000": "5000"})

@task
def run_tests():
    # Run the tests here
    # You can use 'pytest' or 'unittest' to run the tests
    # If the tests pass successfully, this task will complete successfully
    # If any of the tests fail, an exception will be raised, and the pipeline will stop

with Flow("Data Pipeline") as flow:
    # Load Data
    data_path = "path_to_your_data.csv"  # Replace with the actual data path
    start_date = "2022-01-01"
    end_date = "2022-01-31"
    loaded_data = load_data(data_path, start_date, end_date)

    # Data Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(loaded_data)

    # Train the Baseline Model
    baseline_model = train_baseline_model(X_train, y_train)

    # Hyperparameter Optimization
    hyperparameter_optimization = optimize_hyperparameters(X_train, y_train)

    # Run Tests
    run_tests_task = run_tests()

    # Build and Deploy Docker Image
    build_image = build_docker_image()
    deploy_image = deploy_docker_image()

# Set up flow dependencies
run_tests_task.set_upstream(X_train)
deploy_image.set_upstream(build_image)
build_image.set_upstream(hyperparameter_optimization)
hyperparameter_optimization.set_upstream(baseline_model)
baseline_model.set_upstream(X_train)
X_train.set_upstream(loaded_data)



