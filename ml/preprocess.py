# This file handles feature scaling, selection and class imbalance before training

import pandas as pd  # data manipulation
import numpy as np  # numerical operations
from sqlalchemy import create_engine  # database connection
from sklearn.preprocessing import StandardScaler  # scale numerical features
from sklearn.feature_selection import SelectKBest, f_classif  # select top k important features
from imblearn.over_sampling import SMOTE  # handle class imbalance by oversampling minority class
import pickle  # save scaler and selector to disk
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

def load_transformed_data():
    try:
        engine = get_pg_engine()  # get postgresql connection
        df = pd.read_sql("SELECT * FROM transformed_shoppers", engine)  # fetch transformed data
        logger.info(f"Loaded {len(df)} rows from PostgreSQL warehouse")  # log count
        return df  # return dataframe
    except Exception as e:
        logger.error(f"Failed to load transformed data: {e}")  # log error
        raise  # re-raise exception

def preprocess(df):
    try:
        logger.info("Starting preprocessing")  # log start

        X = df.drop('Revenue', axis=1)  # features
        y = df['Revenue']  # target

        # check class distribution before SMOTE
        logger.info(f"Before SMOTE — Revenue 0: {(y==0).sum()}, Revenue 1: {(y==1).sum()}")

        # scale numerical features to same range (mean=0, std=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # fit and transform features
        logger.info("Feature scaling complete")  # log success

        # select top 10 most important features using ANOVA f-test
        selector = SelectKBest(score_func=f_classif, k=10)
        X_selected = selector.fit_transform(X_scaled, y)  # fit and select features

        # get selected feature names for reference
        selected_features = X.columns[selector.get_support()].tolist()
        logger.info(f"Selected Features: {selected_features}")  # log selected features

        # apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_selected, y)  # oversample minority class
        logger.info(f"After SMOTE — Revenue 0: {(y_resampled==0).sum()}, Revenue 1: {(y_resampled==1).sum()}")

        # save scaler to disk for use in prediction
        os.makedirs('./ml', exist_ok=True)  # create ml dir if not exists
        with open('./ml/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)  # serialize scaler
        logger.info("Scaler saved to ./ml/scaler.pkl")  # log save

        # save selector to disk for use in prediction
        with open('./ml/selector.pkl', 'wb') as f:
            pickle.dump(selector, f)  # serialize selector
        logger.info("Selector saved to ./ml/selector.pkl")  # log save

        # save selected feature names for reference in prediction
        with open('./ml/selected_features.pkl', 'wb') as f:
            pickle.dump(selected_features, f)  # serialize feature names
        logger.info("Selected features saved to ./ml/selected_features.pkl")  # log save

        logger.info(f"Preprocessing complete — final shape: {X_resampled.shape}")  # log final shape
        return X_resampled, y_resampled  # return preprocessed data

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")  # log error
        raise  # re-raise exception

if __name__ == "__main__":
    df = load_transformed_data()  # load data from postgresql
    X, y = preprocess(df)  # run preprocessing
    print(f"Final shape after preprocessing: {X.shape}")  # log final shape