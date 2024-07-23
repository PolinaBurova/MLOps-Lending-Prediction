import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
from pydantic import BaseModel
import json
import pandas as pd



# define the inputs we expect our API to receive in the request body as JSON
class Request(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# create an app
app = fastapi.FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# app function is initiated
@app.on_event("startup")
async def startup_event():
    # create a set of tracking URI to appoint the database file 
    # where the metadata is registered
    with open(r'C:\Users\polin\Downloads\projecto_final_OML\projecto_final\rumos_bank\config\app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(config["tracking_uri"])


# predict endpoint which will be called to receive the requests with the inputs of the model 
# and will return a prediction of the model 
@app.post("/predict")
async def root(input: Request):  
    # read app config
    with open(r'C:\Users\polin\Downloads\projecto_final_OML\projecto_final\rumos_bank\config\app.json') as f:
        config = json.load(f)
    # load registered model and its version
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{config['model_name']}/{config['model_version']}"
    )
    # create a dataframe with model's input that we received in the request 
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.dict().items()})
    # call the predict function and have its prediction 
    prediction = model.predict(input_df)
    # return a dictionary format answer with key associated to prediction 
    return {"prediction": prediction.tolist()[0]}


uvicorn.run(app=app, port=5006)
