from joblib import load
import pandas as pd 
from tools import clean_data
from pydantic import BaseModel
from typing import Union
from fastapi import FastAPI, File, UploadFile

import warnings
warnings.filterwarnings("ignore")

description = """
Investissors API aims to provide a classifier to predict 
that a player is worth investing in because he'll last more than 5 years in the NBA, based on his 
sports statistics. The goal is to advise investors looking to 
capitalize on future NBA talent.

## Preview

Where you can: 
* `/preview` a few rows of your dataset


## Machine-Learning 

Where you can:
* `/predict` if this player is likely to last more than 5 years in NBA 
* `/batch-predict` where you can upload a file to get predictions for several players


Check out documentation for more information on each endpoint. 
"""

tags_metadata = [
    
    {
        "name": "Preview",
        "description": "Endpoints that quickly explore dataset"
    },

    {
        "name": "Predictions",
        "description": "Endpoints that uses our Machine Learning model for prediction"
    }
]

app = FastAPI(
    title="👨‍💼 Ce joueur va durer plus de 5 ans en NBA ? - API",
    description=description,
    version="0.1",
    contact={
        "name": "Investissors API - by Kenneth",
    },
    openapi_tags=tags_metadata
)

df = pd.read_csv("nba_logreg.csv")
df_cleaned = clean_data(df)

class PredictionFeatures(BaseModel):
   # PLAYER_NAME: str
    GP: Union[int, float]
    MIN: Union[int, float]
    PTS: Union[int, float]
    FGM: Union[int, float]
    FGA: Union[int, float]
    P3_Made: Union[int, float]
    P3A: Union[int, float]
    FTM: Union[int, float]
    FTA: Union[int, float]
    OREB: Union[int, float]
    DREB: Union[int, float]
    AST: Union[int, float]
    STL: Union[int, float]
    BLK: Union[int, float]
    TOV: Union[int, float]

    # Calculate 3P%
    def P3P(self):
        if self.P3A > 0:
            return (self.P3_Made / self.P3A) * 100
        return df_cleaned['3P%'].mean(skipna=True) 

    # Calculate FT%
    def FTP(self):
        if self.FTA > 0:
            return (self.FTM / self.FTA) * 100
        return df_cleaned['FT%'].mean(skipna=True)

    # Calculate FG%
    def FGP(self):
        if self.FGA > 0:
            return (self.FGM / self.FGA) * 100
        return df_cleaned['FG%'].mean(skipna=True)

    #Calculate REB
    def REB(self):
        return (self.DREB + self.OREB)
   



@app.get("/preview", tags=["Preview"])
async def random_nba_players(rows: int=10):
    """
    It gives you an overview of the dataset. 
    """
    sample = df_cleaned.sample(rows)
    return sample.to_json()


@app.post("/predict", tags=["Machine-Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    Prediction for one observation. Endpoint will return a dictionnary like this:

    ```
    {'prediction': PREDICTION_VALUE[0,1]}
    ```

    You need to give this endpoint all columns values as dictionnary, or form data.
    """
    # Read data 
    prediction = dict(predictionFeatures)
    prediction['3P%'] = PredictionFeatures.P3P
    prediction['FP%'] =  PredictionFeatures.FTP
    prediction['FG%'] =  PredictionFeatures.FGP
    prediction['REB'] =  PredictionFeatures.REB
    df = pd.DataFrame(prediction, index=[0])


    # Load model as a PyFuncModel.
    loaded_model = load('full_pipeline.joblib')
    prediction = loaded_model.predict(df)

    # Format response
    response = {"prediction": prediction.tolist()[0]}
    return response



@app.post("/batch-predict", tags=["Machine-Learning"])
async def batch_predict(file: UploadFile = File(...)):
   
    """ 
   
   Make prediction on a batch of observations. This endpoint accepts only **csv files** containing 
    all the trained columns WITHOUT the target variable 

    """

    df = pd.read_csv(file.file)
    # Load model as a PyFuncModel.
    loaded_model = load('full_pipeline.joblib')
    predictions = loaded_model.predict(df)

    return predictions.tolist()


#if __name__=="__main__":
 #   uvicorn.run(app,host="0.0.0.0", port=4000)