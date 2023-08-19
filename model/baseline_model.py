# model/baseline_model.py
import os,logging,mlflow
from config.config_handler import load_config
from config.logger import LoggerSingleton
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

cfg = load_config()
logger = LoggerSingleton().get_logger()  # Create the logger instance
logging.getLogger().propagate = False

data_dir = cfg.data.data_dir

MODEL_ARTIFACTS_DIR = cfg.models.model_artifact_dir
if not os.path.exists(MODEL_ARTIFACTS_DIR):
    os.makedirs(MODEL_ARTIFACTS_DIR)

def plot_confusion_matrix(actual, predictions,cm_file):
    # Calculate the confusion matrix
    cm = confusion_matrix(actual, predictions)

    # Plot the confusion matrix
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 12})  # Change font size here
    plt.xlabel("Predicted Labels", fontsize=12)  # Change x-label font size
    plt.ylabel("True Labels", fontsize=12)  # Change y-label font size
    plt.title("Confusion Matrix", fontsize=13)  # Change title font size
    plt.xticks(fontsize=12)  # Change x tick labels font size
    plt.yticks(fontsize=12)  # Change y tick labels font size
    plt.tight_layout()
    cm_path = os.path.join(MODEL_ARTIFACTS_DIR, cm_file)
    logging.info("cm_path ::  %s",cm_path)
    plt.savefig(cm_path)
    # plt.show()
    plt.close()
    return cm_path


def calculate_scores(actual, predictions):
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions)
    recall = recall_score(actual, predictions)
    f1 = f1_score(actual, predictions)

    logging.info("Accuracy: %s", accuracy)
    logging.info("F1-Score: %s", f1)
    logging.info("Precision: %s", precision)
    logging.info("Recall: %s", recall)
    
    return accuracy, precision, recall, f1
    

def train_evaluate_LR(X_train, y_train, X_val, y_val):
    try:
        # Create a logistic regression model
        logger.info("Training baseline model")
        lr = LogisticRegression()

        # Train the model on the training data
        lr.fit(X_train, y_train.values.ravel())  # Convert y_train to 1d array using .values.ravel()

        # Make predictions on the validation data
        val_predictions = lr.predict(X_val)

        # Calculate accuracy and F1-score on the validation data
        # Convert y_val to 1d array using .values.ravel()
        accuracy, precision, recall, f1= calculate_scores(y_val.values.ravel(), val_predictions)

        # Get or create the experiment
        experiment_name = cfg.models.baseline_model.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Start an MLflow run to log the model metrics
        with mlflow.start_run(run_name="Baseline_LR_Model", experiment_id=experiment_id):
            # Log the parameters of the logistic regression model
            mlflow.log_params(lr.get_params())

            # Log the evaluation metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            # Log the confusion matrix plot as an artifact
            cm_file = cfg.models.baseline_model.cm_filename
            cm = plot_confusion_matrix(y_val.values.ravel(), val_predictions, cm_file)
            mlflow.log_artifact(cm,  artifact_path="Baseline_CF")

        return accuracy, precision, recall, f1

    except Exception as e:
        logging.exception("An error occurred during model training and evaluation: %s", str(e))
        return None
