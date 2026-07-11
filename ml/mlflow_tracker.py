# This file handles MLflow experiment tracking and model registry for all models

import mlflow  # mlflow tracking library
import mlflow.sklearn  # mlflow sklearn integration
from mlflow.tracking import MlflowClient  # mlflow client for model registry
from dotenv import load_dotenv  # load env variables
import os  # access env variables
import sys  # system path manipulation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add root to path
from utils.logger import get_logger  # import common logger

load_dotenv()  # load .env file

logger = get_logger(__name__)  # get logger for this module

def init_mlflow():
    try:
        # set mlflow tracking uri where runs will be stored
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        # set experiment name — creates new if not exists
        mlflow.set_experiment("shoppers_intent_classification")
        logger.info("MLflow initialized")  # log init
    except Exception as e:
        logger.error(f"MLflow initialization failed: {e}")  # log error
        raise  # re-raise exception

def log_single_run(model, model_name, metrics, is_best=False):
    try:
        with mlflow.start_run(run_name=model_name):  # start mlflow run
            # log model type and whether it is best
            mlflow.log_param("model_type", model_name)          # log model name
            mlflow.log_param("is_best_model", is_best)          # log if best model
            mlflow.log_param("n_estimators", getattr(model, 'n_estimators', 'N/A'))  # log estimators

            # log all evaluation metrics
            mlflow.log_metric("accuracy", metrics['acc'])        # log accuracy
            mlflow.log_metric("precision", metrics['pre'])       # log precision
            mlflow.log_metric("recall", metrics['rec'])          # log recall
            mlflow.log_metric("f1_score", metrics['f1'])         # log f1 score
            mlflow.log_metric("roc_auc", metrics['auc'])         # log roc auc

            # log model artifact to mlflow
            mlflow.sklearn.log_model(model, model_name)          # save model in mlflow

            # register best model in mlflow model registry
            if is_best:
                run_id = mlflow.active_run().info.run_id  # get current run id
                model_uri = f"runs:/{run_id}/{model_name}"  # build model uri
                mlflow.register_model(model_uri, "shoppers_intent_best_model")  # register model
                logger.info(f"Best model '{model_name}' registered in MLflow Model Registry")

            logger.info(f"MLflow run logged for {model_name} — Best: {is_best}")  # log success

    except Exception as e:
        logger.error(f"MLflow logging failed for {model_name}: {e}")  # log error
        raise  # re-raise exception

def log_all_runs(models: dict, best_name: str):
    try:
        init_mlflow()  # initialize mlflow once

        # log each model run
        for model_name, (model, metrics) in models.items():
            is_best = (model_name == best_name)  # check if this is best model
            log_single_run(model, model_name, metrics, is_best=is_best)  # log run

        logger.info(f"All models logged to MLflow — Best: {best_name}")  # log completion

    except Exception as e:
        logger.error(f"Failed to log all runs to MLflow: {e}")  # log error
        raise  # re-raise exception