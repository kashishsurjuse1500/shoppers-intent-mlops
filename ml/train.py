# This file trains RF, XGBoost, LightGBM models with hyperparameter tuning and selects best model

import pandas as pd  # data manipulation
from sklearn.model_selection import train_test_split, GridSearchCV  # split data and hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # random forest model
from xgboost import XGBClassifier  # xgboost model
from lightgbm import LGBMClassifier  # lightgbm model
from sklearn.metrics import (
    accuracy_score,    # overall accuracy
    precision_score,   # precision score
    recall_score,      # recall score
    f1_score,          # f1 score
    roc_auc_score      # roc auc score
)
import pickle  # save model to disk
from dotenv import load_dotenv  # load env variables
import os  # access env variables
import sys  # system path manipulation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add root to path
from utils.logger import get_logger  # import common logger
from preprocess import load_transformed_data, preprocess  # import preprocess functions
from mlflow_tracker import log_all_runs  # import mlflow logging function

load_dotenv()  # load .env file

logger = get_logger(__name__)  # get logger for this module

# hyperparameter grids for each model
RF_PARAMS = {
    'n_estimators': [100, 200],       # number of trees
    'max_depth': [None, 10, 20],      # max depth of trees
    'min_samples_split': [2, 5]       # min samples to split node
}

XGB_PARAMS = {
    'n_estimators': [100, 200],       # number of boosting rounds
    'max_depth': [3, 6],              # max depth of trees
    'learning_rate': [0.01, 0.1]      # learning rate
}

LGBM_PARAMS = {
    'n_estimators': [100, 200],       # number of boosting rounds
    'max_depth': [-1, 10],            # max depth (-1 means no limit)
    'learning_rate': [0.01, 0.1]      # learning rate
}

def tune_model(model, params, X_train, y_train, model_name):
    try:
        logger.info(f"Starting hyperparameter tuning for {model_name}")  # log start
        grid_search = GridSearchCV(
            model,           # model to tune
            params,          # parameter grid
            cv=3,            # 3-fold cross validation
            scoring='roc_auc',  # optimize for roc auc
            n_jobs=-1        # use all cpu cores
        )
        grid_search.fit(X_train, y_train)  # fit grid search
        logger.info(f"{model_name} best params: {grid_search.best_params_}")  # log best params
        logger.info(f"{model_name} best cv score: {grid_search.best_score_:.4f}")  # log best score
        return grid_search.best_estimator_  # return best model
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed for {model_name}: {e}")  # log error
        raise  # re-raise exception

def get_metrics(model, X_test, y_test, model_name):
    try:
        y_pred = model.predict(X_test)  # predictions
        y_prob = model.predict_proba(X_test)[:, 1]  # probabilities
        metrics = {
            'acc': accuracy_score(y_test, y_pred),           # accuracy
            'pre': precision_score(y_test, y_pred),           # precision
            'rec': recall_score(y_test, y_pred),              # recall
            'f1':  f1_score(y_test, y_pred),                  # f1 score
            'auc': roc_auc_score(y_test, y_prob)              # roc auc
        }
        logger.info(f"{model_name} — Accuracy: {metrics['acc']:.4f} | AUC: {metrics['auc']:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics for {model_name}: {e}")  # log error
        raise  # re-raise exception

def train():
    try:
        logger.info("Training pipeline started")  # log start

        df = load_transformed_data()  # load transformed data from postgresql
        X, y = preprocess(df)  # apply scaling, feature selection and SMOTE

        # split into 80% train and 20% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")  # log split

        # tune and train random forest
        rf_model = tune_model(RandomForestClassifier(random_state=42), RF_PARAMS, X_train, y_train, "RandomForest")
        rf_metrics = get_metrics(rf_model, X_test, y_test, "RandomForest")  # get rf metrics

        # tune and train xgboost
        xgb_model = tune_model(XGBClassifier(random_state=42, eval_metric='logloss'), XGB_PARAMS, X_train, y_train, "XGBoost")
        xgb_metrics = get_metrics(xgb_model, X_test, y_test, "XGBoost")  # get xgb metrics

        # tune and train lightgbm
        lgbm_model = tune_model(LGBMClassifier(random_state=42, verbose=-1), LGBM_PARAMS, X_train, y_train, "LightGBM")
        lgbm_metrics = get_metrics(lgbm_model, X_test, y_test, "LightGBM")  # get lgbm metrics

        # select best model based on roc auc score
        models = {
            "RandomForest": (rf_model, rf_metrics),
            "XGBoost": (xgb_model, xgb_metrics),
            "LightGBM": (lgbm_model, lgbm_metrics)
        }
        best_name = max(models, key=lambda x: models[x][1]['auc'])  # select best by auc
        best_model, best_metrics = models[best_name]  # get best model and metrics
        logger.info(f"Best Model: {best_name} — AUC: {best_metrics['auc']:.4f}")  # log best model

        # log all models to mlflow and register best model
        log_all_runs(
            models={
                "RandomForest": (rf_model, rf_metrics),
                "XGBoost": (xgb_model, xgb_metrics),
                "LightGBM": (lgbm_model, lgbm_metrics)
            },
            best_name=best_name  # pass best model name for registry
        )

        # save best model to disk
        os.makedirs('./ml', exist_ok=True)  # create ml dir if not exists
        with open(os.getenv('MODEL_PATH'), 'wb') as f:
            pickle.dump(best_model, f)  # serialize model
        logger.info(f"Best model saved to {os.getenv('MODEL_PATH')}")  # log save path

        return best_model, X_test, y_test, best_name  # return for evaluate use

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")  # log error
        raise  # re-raise exception

if __name__ == "__main__":
    train()  # run trainings