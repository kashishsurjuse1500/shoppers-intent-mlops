# This file loads CSV data into MySQL with correct TRUE/FALSE conversion
# This file loads CSV data into MySQL with correct TRUE/FALSE conversion

import pandas as pd  # data manipulation
from sqlalchemy import create_engine  # database connection
from dotenv import load_dotenv  # load env variables
import os  # access env variables
import sys  # system path manipulation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add root to path
from utils.logger import get_logger  # import common logger

load_dotenv()  # load .env file

logger = get_logger(__name__)  # get logger for this module

def get_engine():
    try:
        # build mysql connection string
        url = (
            f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        engine = create_engine(url)  # return engine
        logger.info("MySQL engine created successfully")  # log success
        return engine
    except Exception as e:
        logger.error(f"Failed to create MySQL engine: {e}")  # log error
        raise  # re-raise exception

def reload():
    try:
        engine = get_engine()  # get db connection
        df = pd.read_csv("data/online_shoppers_intention.csv")  # read csv file
        logger.info(f"CSV loaded — {len(df)} rows")  # log count

        # convert Weekend and Revenue to int — handle multiple formats
        df['Weekend'] = df['Weekend'].astype(str).str.strip().str.upper().map({'TRUE': 1, 'FALSE': 0})
        df['Revenue'] = df['Revenue'].astype(str).str.strip().str.upper().map({'TRUE': 1, 'FALSE': 0})
        logger.info("Weekend and Revenue columns converted successfully")  # log success

        df.to_sql('online_shoppers', engine, if_exists='replace', index=False)  # load to mysql
        logger.info(f"Loaded {len(df)} rows into MySQL")  # log count
        logger.info(f"Revenue distribution:\n{df['Revenue'].value_counts()}")  # log distribution

    except Exception as e:
        logger.error(f"Data reload failed: {e}")  # log error
        raise  # re-raise exception

if __name__ == "__main__":
    reload()  # run reload
