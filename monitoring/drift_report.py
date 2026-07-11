# This file generates data drift report using Evidently AI from PostgreSQL warehouse

import pandas as pd  # data manipulation
from evidently.report import Report  # evidently report class
from evidently.metric_preset import DataDriftPreset  # preset for data drift metrics
from sqlalchemy import create_engine  # database connection
from dotenv import load_dotenv  # load env variables
import os  # access env variables
import sys  # system path manipulation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add root to path
from utils.logger import get_logger  # import common logger

load_dotenv()  # load .env file

logger = get_logger(__name__)  # get logger for this module

def get_pg_engine():
    try:
        # build postgresql connection string for data warehouse
        url = (
            f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
            f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_NAME')}"
        )
        engine = create_engine(url)  # return postgresql engine
        logger.info("PostgreSQL engine created successfully")  # log success
        return engine
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL engine: {e}")  # log error
        raise  # re-raise exception

def generate_drift_report():
    try:
        logger.info("Drift report generation started")  # log start

        engine = get_pg_engine()  # get postgresql warehouse connection

        # load full transformed data from postgresql warehouse
        df = pd.read_sql("SELECT * FROM transformed_shoppers", engine)
        logger.info(f"Loaded {len(df)} rows from PostgreSQL warehouse")  # log count

        # use first 70% as reference (training data)
        reference = df.iloc[:int(len(df) * 0.7)]
        logger.info(f"Reference size: {len(reference)} rows")  # log reference size

        # use last 30% as current (new incoming data)
        current = df.iloc[int(len(df) * 0.7):]
        logger.info(f"Current size: {len(current)} rows")  # log current size

        # create evidently report with data drift preset
        report = Report(metrics=[DataDriftPreset()])

        # run report on reference vs current data
        report.run(reference_data=reference, current_data=current)

        # save report as html file
        os.makedirs('./monitoring', exist_ok=True)  # create monitoring dir if not exists
        report.save_html('./monitoring/drift_report.html')  # save html report
        logger.info("Drift report saved to ./monitoring/drift_report.html")  # log save

    except Exception as e:
        logger.error(f"Drift report generation failed: {e}")  # log error
        raise  # re-raise exception

if __name__ == "__main__":
    generate_drift_report()  # run drift report generation