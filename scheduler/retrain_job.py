# This file schedules automatic ETL + model retraining using APScheduler

from apscheduler.schedulers.blocking import BlockingScheduler  # blocking scheduler runs forever
from apscheduler.triggers.cron import CronTrigger  # cron trigger for scheduling
import sys  # system path manipulation
import os  # access env variables
from dotenv import load_dotenv  # load env variables

load_dotenv()  # load .env file

# add root, ml and etl directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../etl')))

from utils.logger import get_logger  # import common logger
from etl.extract import extract_data      # import extract function
from etl.transform import transform_data  # import transform function
from etl.load import load_data            # import load function
from ml.train import train                # import train function
from monitoring.drift_report import generate_drift_report  # import drift report function

logger = get_logger(__name__)  # get logger for this module

def run_etl():
    try:
        logger.info("ETL pipeline started")  # log start
        df = extract_data()           # extract raw data from mysql
        df = transform_data(df)       # clean and encode data
        load_data(df)                 # load into postgresql warehouse
        logger.info("ETL pipeline completed")  # log completion
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")  # log error
        raise  # re-raise exception

def run_training():
    try:
        logger.info("Training pipeline started")  # log start
        train()                       # train model and log to mlflow
        logger.info("Training pipeline completed")  # log completion
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")  # log error
        raise  # re-raise exception

def run_drift_report():
    try:
        logger.info("Drift report started")  # log start
        generate_drift_report()       # generate drift report
        logger.info("Drift report completed")  # log completion
    except Exception as e:
        logger.error(f"Drift report failed: {e}")  # log error
        raise  # re-raise exception

def run_pipeline():
    try:
        logger.info("Full pipeline started")  # log start
        run_etl()          # run ETL first
        run_training()     # then retrain model
        run_drift_report() # then generate drift report
        logger.info("Full pipeline completed")  # log completion
    except Exception as e:
        logger.error(f"Full pipeline failed: {e}")  # log error
        raise  # re-raise exception

if __name__ == "__main__":
    scheduler = BlockingScheduler()  # create scheduler instance

    # schedule pipeline to run every day at 2:00 AM
    scheduler.add_job(
        run_pipeline,                   # function to run
        CronTrigger(hour=2, minute=0),  # every day at 2 AM
        id='retrain_job',               # unique job id
        name='ETL + Retrain Job',       # job name
        replace_existing=True           # replace if job already exists
    )

    logger.info("Scheduler started — pipeline will run daily at 2:00 AM")  # log start

    try:
        scheduler.start()          # start scheduler — runs forever
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")  # log stop on ctrl+c