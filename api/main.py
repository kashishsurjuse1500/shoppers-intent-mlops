# This file defines FastAPI app with prediction, health check and frontend endpoints

from fastapi import FastAPI, HTTPException  # fastapi framework
from fastapi.staticfiles import StaticFiles  # serve static files
from fastapi.templating import Jinja2Templates  # serve html templates
from fastapi.requests import Request  # request object
from fastapi.responses import HTMLResponse  # html response
from pydantic import BaseModel  # request body validation
import sys  # system path manipulation
import os  # access env variables
from dotenv import load_dotenv  # load env variables

load_dotenv()  # load .env file

# add ml directory to path so predict imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml')))

from ml.predict import predict  # import predict function
from utils.logger import get_logger  # import common logger

logger = get_logger(__name__)  # get logger for this module

app = FastAPI(
    title="Online Shoppers Intention API",  # api title
    description="Predicts if a shopper will purchase or not",  # api description
    version="2.0.0"  # api version
)

# mount static files directory
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# setup jinja2 templates
templates = Jinja2Templates(directory="api/templates")

# define input schema using pydantic — all 17 features
class ShopperInput(BaseModel):
    Administrative: int              # number of administrative pages visited
    Administrative_Duration: float   # time spent on administrative pages
    Informational: int               # number of informational pages visited
    Informational_Duration: float    # time spent on informational pages
    ProductRelated: int              # number of product related pages visited
    ProductRelated_Duration: float   # time spent on product related pages
    BounceRates: float               # bounce rate of visited pages
    ExitRates: float                 # exit rate of visited pages
    PageValues: float                # page value of visited pages
    SpecialDay: float                # closeness to special day
    Month: int                       # month of visit encoded as int
    OperatingSystems: int            # operating system of visitor
    Browser: int                     # browser of visitor
    Region: int                      # region of visitor
    TrafficType: int                 # traffic type of visitor
    VisitorType: int                 # visitor type encoded as int
    Weekend: int                     # 1 if weekend else 0

# define output schema
class PredictionOutput(BaseModel):
    prediction: int      # 0 or 1
    probability: float   # confidence score
    message: str         # human readable result

@app.get("/", response_class=HTMLResponse)  # serve frontend
def frontend(request: Request):
    try:
        logger.info("Frontend requested")  # log request
        return templates.TemplateResponse(request, "index.html")  # render html
    except Exception as e:
        logger.error(f"Frontend failed: {e}")  # log error
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")  # health check endpoint
def health():
    logger.info("Health check requested")  # log request
    return {"status": "healthy"}  # return healthy status

@app.post("/predict", response_model=PredictionOutput)  # prediction endpoint
def predict_endpoint(data: ShopperInput):
    try:
        logger.info(f"Prediction requested")  # log request
        input_dict = data.model_dump()  # convert pydantic model to dict
        result = predict(input_dict)    # run prediction
        logger.info(f"Prediction result: {result['message']}")  # log result
        return result                   # return prediction result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")  # log error
        raise HTTPException(status_code=500, detail=str(e))