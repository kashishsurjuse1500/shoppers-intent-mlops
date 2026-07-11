# This file extracts raw data from MySQL database

import pandas as pd  # data manipulation library
from sqlalchemy import create_engine  # database connection engine
from dotenv import load_dotenv  # load environment variables from .env file
import os  # access environment variables
import sys  # system path manipulation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add root to path
from utils.logger import get_logger  # import common logger

load_dotenv()  # load .env file variables into environment

logger = get_logger(__name__)  # get logger for this module

def get_engine():
    try:
        # build mysql connection string using env variables
        url = (
            f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        engine = create_engine(url)  # return sqlalchemy engine object
        logger.info("MySQL engine created successfully")  # log success
        return engine
    except Exception as e:
        logger.error(f"Failed to create MySQL engine: {e}")  # log error
        raise  # re-raise exception

def extract_data():
    try:
        engine = get_engine()  # get database connection
        df = pd.read_sql("SELECT * FROM online_shoppers", engine)  # fetch all rows from table
        logger.info(f"Extracted {len(df)} rows from MySQL")  # log row count
        return df  # return dataframe
    except Exception as e:
        logger.error(f"Failed to extract data: {e}")  # log error
        raise  # re-raise exception

if __name__ == "__main__":
    df = extract_data()  # run extraction
    print(df.head())  # print first 5 rows to verifys