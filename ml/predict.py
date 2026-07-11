# This file loads saved model and makes predictions on new input data

import pickle  # load saved model, scaler, selector
import numpy as np  # numerical operations
import pandas as pd  # data manipulation
from dotenv import load_dotenv  # load env variables
import os  # access env variables
import sys  # system path manipulation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add root to path
from utils.logger import get_logger  # import common logger

load_dotenv()  # load .env file

logger = get_logger(__name__)  # get logger for this module

def load_artifacts():
    try:
        # load saved model from disk
        with open(os.getenv('MODEL_PATH'), 'rb') as f:
            model = pickle.load(f)  # deserialize model
        logger.info("Model loaded successfully")  # log success

        # load saved scaler from disk
        with open('./ml/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)  # deserialize scaler
        logger.info("Scaler loaded successfully")  # log success

        # load saved selector from disk
        with open('./ml/selector.pkl', 'rb') as f:
            selector = pickle.load(f)  # deserialize selector
        logger.info("Selector loaded successfully")  # log success

        # load selected feature names
        with open('./ml/selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)  # deserialize feature names
        logger.info("Selected features loaded successfully")  # log success

        return model, scaler, selector, selected_features  # return all artifacts

    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")  # log error
        raise  # re-raise exception

def predict(input_data: dict):
    try:
        model, scaler, selector, selected_features = load_artifacts()  # load all artifacts

        # convert input dict to dataframe
        df = pd.DataFrame([input_data])

        # scale ALL 17 features first (same as training)
        df_scaled = scaler.transform(df)
        logger.info("Input data scaled successfully")  # log success

        # then select top 10 features using saved selector
        df_selected = selector.transform(df_scaled)
        logger.info("Feature selection applied successfully")  # log success

        # make prediction
        prediction = model.predict(df_selected)[0]  # 0 or 1
        probability = model.predict_proba(df_selected)[0][1]  # probability of Revenue=1

        result = {
            "prediction": int(prediction),  # 0 = no purchase, 1 = purchase
            "probability": round(float(probability), 4),  # confidence score
            "message": "Will Purchase" if prediction == 1 else "Will Not Purchase"  # human readable
        }

        logger.info(f"Prediction: {result['message']} — Probability: {result['probability']}")
        return result  # return prediction result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")  # log error
        raise  # re-raise exception

if __name__ == "__main__":
    # sample input — all 17 features
    sample_input = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 0.0,
        "BounceRates": 0.2,
        "ExitRates": 0.2,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": 2,
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": 2,
        "Weekend": 0
    }

    result = predict(sample_input)  # run prediction
    print(result)  # print result