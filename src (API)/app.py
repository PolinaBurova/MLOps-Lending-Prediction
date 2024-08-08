import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
from pydantic import BaseModel
import json
import pandas as pd



# Define the inputs that we expect our API to receive in the body of the request as JSON.
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



# Function that is called when the app is started.
@app.on_event("startup")
async def startup_event():
    # We set the tracking URI to point to the db file
    # where the metadata of our registered models is stored

    with open(r'C:\rumos_bank\...config\app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(config["tracking_uri"])


# Predict endpoint that will be called to receive requests with the inputs for the model
# and will return the model's prediction in the response

@app.post("/predict")
async def root(input: Request):  
    # read app config
    with open(r'C:\Users\...config\app.json') as f:
        config = json.load(f)
    # We load the registered model
    # according to the model name and model version read from the config

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{config['model_name']}/{config['model_version']}"
    )
    # We construct a DataFrame with the model inputs received in the request
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.dict().items()})
    # # We call the predict function of the model and obtain its prediction
    prediction = model.predict(input_df)
    # We return as a response a dictionary with the prediction associated with the key "prediction"

    return {"prediction": prediction.tolist()[0]}


uvicorn.run(app=app, port=5007)
