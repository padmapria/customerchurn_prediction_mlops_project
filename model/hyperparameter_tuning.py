# model/hyperparameter_tuning.py
import os,logging,joblib,mlflow,json
from config.config_handler import load_config
from config.logger import LoggerSingleton
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from model.baseline_model import calculate_scores,plot_confusion_matrix

# Load the model configuration from config.config_handler
cfg = load_config()
logger = LoggerSingleton().get_logger()  # Create the logger instance
logging.getLogger().propagate = False

MODEL_ARTIFACTS_DIR = cfg.models.model_artifact_dir

def perform_grid_search(classifier, param_grid, X_train, y_train, X_val, y_val,
                        model_filename, params_filename, cm_filename,experiment_name, register_model=False):
    logger.info(f"Gridsearch {classifier}")

    # Convert y_train and y_val to 1D arrays using ravel()
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()

    # Grid Search Cross Validation
    grid_search = GridSearchCV(classifier, param_grid, cv=5)

    experiment_name = cfg.models.baseline_model.experiment_name
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Start MLflow run for the entire grid search
    with mlflow.start_run(run_name=f"{classifier}_classifier",experiment_id=experiment_id) as grid_search_run:
        mlflow.log_params(param_grid)

        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Initialize variables to keep track of the best model and its performance
        best_model = None
        best_accuracy = 0
        best_params = None

        # Log each individual model run in the grid search loop
        for i, params in enumerate(grid_search.cv_results_['params']):
            model = classifier.set_params(**params)
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)

            gs_accuracy, gs_precision, gs_recall, gs_f1 = calculate_scores(y_val, val_predictions)

            mlflow.log_metrics({
                "accuracy": gs_accuracy,
                "precision": gs_precision,
                "recall": gs_recall,
                "f1": gs_f1
            })

            # Check if this model is the best-performing so far
            if gs_recall > best_accuracy:
                best_accuracy = gs_recall
                best_model = model
                best_params = params
                predictions = val_predictions

        # Save the best model for the entire grid search as an artifact
        if best_model is not None:
            model_path = os.path.join(MODEL_ARTIFACTS_DIR, model_filename)
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(model_path)

            cm = plot_confusion_matrix(y_val, val_predictions, cm_filename)
            mlflow.log_artifact(cm, cm_filename)

            # Register the best model under a separate experiment name
            if(register_model):
                mlflow.set_experiment(cfg.mlflow.best_artifact_experiment_name)
                mlflow.register_model(model_uri=f"runs:/{grid_search_run.info.run_id}/{model_filename}",
                                      name=model_filename)

        # Save the best parameters as an artifact
        if best_params is not None:
            params_path = os.path.join(MODEL_ARTIFACTS_DIR, params_filename)
            with open(params_path, "w") as f:
                json.dump(best_params, f)
            mlflow.log_artifact(params_path)


def grid_search_RF(X_train, y_train, X_val, y_val):
    rf_param_grid = {
        'n_estimators': cfg.models.random_forest.model_parameters.param_grid.n_estimators,
        'max_depth': cfg.models.random_forest.model_parameters.param_grid.max_depth,
        'min_samples_split': cfg.models.random_forest.model_parameters.param_grid.min_samples_split
    }

    perform_grid_search(RandomForestClassifier(class_weight='balanced'), rf_param_grid, X_train, y_train, X_val, y_val,
                        cfg.models.random_forest.model_filename, cfg.models.random_forest.params_filename,
                        cfg.models.random_forest.cm_filename,cfg.models.random_forest.experiment_name,True)


def grid_search_GB(X_train, y_train, X_val, y_val):
    gb_param_grid = {
        'n_estimators': cfg.models.gradient_boosting.model_parameters.param_grid.n_estimators,
        'learning_rate': cfg.models.gradient_boosting.model_parameters.param_grid.learning_rate,
        'max_depth': cfg.models.gradient_boosting.model_parameters.param_grid.max_depth
    }
    perform_grid_search(GradientBoostingClassifier(), gb_param_grid, X_train, y_train, X_val, y_val,
                        cfg.models.gradient_boosting.model_filename, cfg.models.gradient_boosting.params_filename,
                        cfg.models.gradient_boosting.cm_filename,cfg.models.gradient_boosting.experiment_name,False)
