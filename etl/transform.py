# This file cleans, validates and preprocesses raw data for ML model

import pandas as pd  # data manipulation library
from sklearn.preprocessing import LabelEncoder  # encode categorical columns to numbers
import os  # access environment variables
import sys  # system path manipulation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add root to path
from utils.logger import get_logger  # import common logger

logger = get_logger(__name__)  # get logger for this module

def validate_data(df):
    # data validation — check schema, types, nulls, value ranges
    try:
        # check required columns exist
        required_columns = [
            'Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
            'OperatingSystems', 'Browser', 'Region', 'TrafficType',
            'VisitorType', 'Weekend', 'Revenue'
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]  # find missing columns
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")  # raise error if missing
        logger.info("Column validation passed")  # log success

        # check null values
        null_counts = df.isnull().sum()  # count nulls per column
        if null_counts.any():
            logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")  # log warning
        else:
            logger.info("Null value validation passed")  # log success

        # check BounceRates and ExitRates are between 0 and 1
        if not df['BounceRates'].between(0, 1).all():
            logger.warning("BounceRates contains values outside 0-1 range")  # log warning
        if not df['ExitRates'].between(0, 1).all():
            logger.warning("ExitRates contains values outside 0-1 range")  # log warning
        logger.info("Value range validation passed")  # log success

        # check Revenue column has only 0 and 1
        if not df['Revenue'].isin([0, 1]).all():
            raise ValueError("Revenue column contains invalid values — expected 0 or 1")  # raise error
        logger.info("Revenue column validation passed")  # log success

        # check row count
        if len(df) == 0:
            raise ValueError("Dataframe is empty — no data to process")  # raise error
        logger.info(f"Row count validation passed — {len(df)} rows")  # log success

    except Exception as e:
        logger.error(f"Data validation failed: {e}")  # log error
        raise  # re-raise exception

def transform_data(df):
    try:
        logger.info("Starting data transformation")  # log start

        # validate data before transformation
        validate_data(df)

        # drop duplicate rows if any
        before = len(df)  # row count before
        df = df.drop_duplicates()  # remove duplicates
        logger.info(f"Dropped {before - len(df)} duplicate rows")  # log dropped count

        # drop rows where any value is null
        before = len(df)  # row count before
        df = df.dropna()  # remove nulls
        logger.info(f"Dropped {before - len(df)} null rows")  # log dropped count

        # encode Month column (Jan, Feb... → 1, 2...)
        le_month = LabelEncoder()
        df['Month'] = le_month.fit_transform(df['Month'])
        logger.info("Month column encoded")  # log success

        # encode VisitorType column (Returning_Visitor, New_Visitor... → 0, 1...)
        le_visitor = LabelEncoder()
        df['VisitorType'] = le_visitor.fit_transform(df['VisitorType'])
        logger.info("VisitorType column encoded")  # log success

        # convert Weekend column to integer (True → 1, False → 0)
        df['Weekend'] = df['Weekend'].astype(int)

        # convert Revenue column to integer (True → 1, False → 0)
        df['Revenue'] = df['Revenue'].astype(int)

        logger.info(f"Transformation complete — final shape: {df.shape}")  # log final shape
        return df  # return cleaned dataframe

    except Exception as e:
        logger.error(f"Data transformation failed: {e}")  # log error
        raise  # re-raise exception

if __name__ == "__main__":
    from extract import extract_data  # import extract function
    df = extract_data()  # fetch raw data
    df = transform_data(df)  # apply transformations
    print(df.head())  # print first 5 rows to verify
    print(df.dtypes)  # print column data types to verify encoding