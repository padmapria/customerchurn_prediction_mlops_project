# model/baseline_model.py

from config.config_handler import load_config
import os,logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

cfg = load_config()

data_dir = cfg.data.data_dir
MODEL_ARTIFACTS_DIR = cfg.models.model_artifact_dir
cm_file = cfg.models.baseline_model.cm_filename

def calculate_scores(actual, predictions,cm_file):
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions)
    recall = recall_score(actual, predictions)
    f1 = f1_score(actual, predictions)
    
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
    plt.savefig(os.path.join(MODEL_ARTIFACTS_DIR , cm_file) )
    #plt.show()
    
    logging.info("Accuracy: %s", accuracy)
    logging.info("F1-Score: %s", f1)
    logging.info("Precision: %s", precision)
    logging.info("Recall: %s", recall)
    
    return accuracy, precision, recall, f1
    

def train_evaluate_LR(X_train, y_train, X_val, y_val):
    try:
        # Create a logistic regression model
        logging.info("Training baseline model")
        lr = LogisticRegression()

        # Train the model on the training data
        lr.fit(X_train, y_train.values.ravel())  # Convert y_train to 1d array using .values.ravel()

        # Make predictions on the validation data
        val_predictions = lr.predict(X_val)

        # Calculate accuracy and F1-score on the validation data
        # Convert y_val to 1d array using .values.ravel()
        return calculate_scores(y_val.values.ravel(), val_predictions, cm_file)

    except Exception as e:
        logging.exception("An error occurred during model training and evaluation: %s", str(e))
        return None
