# model/hyperparameter_tuning.py

from config.config_handler import load_config
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from model.baseline_model import calculate_scores
import os,joblib,logging

# Load the model configuration from config.config_handler
cfg = load_config()

MODEL_ARTIFACTS_DIR = cfg.models.model_artifact_dir
rf_param_grid = {'n_estimators': cfg.models.random_forest.model_parameters.param_grid.n_estimators,
                 'max_depth': cfg.models.random_forest.model_parameters.param_grid.max_depth,
                 'min_samples_split': cfg.models.random_forest.model_parameters.param_grid.min_samples_split}

gb_param_grid = {'n_estimators': cfg.models.gradient_boosting.model_parameters.param_grid.n_estimators,
                 'learning_rate': cfg.models.gradient_boosting.model_parameters.param_grid.learning_rate,
                 'max_depth': cfg.models.gradient_boosting.model_parameters.param_grid.max_depth}



def grid_search_RF(X_train, y_train, X_val, y_val):
    
    logging.info("Gridsearch RF")
    # Create a RandomForestClassifier
    rf = RandomForestClassifier(class_weight='balanced')
    
    # Convert y_train and y_val to 1D arrays using ravel()
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()

    # Grid Search Cross Validation
    grid_search = GridSearchCV(rf, rf_param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    best_params = grid_search.best_params_

    # Model evaluation using the best hyperparameters on the validation data
    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train, y_train)
    val_predictions = best_rf.predict(X_val)

    gs_accuracy, gs_precision, gs_recall, gs_f1 = calculate_scores(y_val, val_predictions,cfg.models.random_forest.cm_filename )

    # Save the best model and parameters to the specified directory
    model_path = os.path.join(MODEL_ARTIFACTS_DIR, cfg.models.random_forest.model_filename)
    params_path = os.path.join(MODEL_ARTIFACTS_DIR, cfg.models.random_forest.params_filename)
    joblib.dump(best_rf, model_path)
    joblib.dump(best_params, params_path)

    # Return the results
    #return best_params, gs_accuracy, gs_precision, gs_recall, gs_f1


def grid_search_GB(X_train, y_train, X_val, y_val):
    # Create a GradientBoostingClassifier
    logging.info("Gridsearch GB")
    
    gb = GradientBoostingClassifier()
    
    # Convert y_train and y_val to 1D arrays using ravel()
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()

    # Grid Search Cross Validation
    grid_search = GridSearchCV(gb, gb_param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    best_params = grid_search.best_params_

    # Model evaluation using the best hyperparameters on the validation data
    best_gb = GradientBoostingClassifier(**best_params)
    best_gb.fit(X_train, y_train)
    val_predictions = best_gb.predict(X_val)

    gs_accuracy, gs_precision, gs_recall, gs_f1 = calculate_scores(y_val, val_predictions,gradient_boosting.cm_filename)

    # Save the best model and parameters to the specified directory
    model_path = os.path.join(MODEL_ARTIFACTS_DIR, cfg.models.gradient_boosting.model_filename)
    params_path = os.path.join(MODEL_ARTIFACTS_DIR, cfg.models.gradient_boosting.params_filename)
    joblib.dump(best_gb, model_path)
    joblib.dump(best_params, params_path)

    # Return the results
    #return best_params, gs_accuracy, gs_precision, gs_recall, gs_f1