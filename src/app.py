import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
from pydantic import BaseModel
import json
import pandas as pd



# definir os inputs que estamos à espera que a nossa API receba no body do pedido como JSON
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


# criar uma app
app = fastapi.FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# função que é chamada quando a app é iniciada
@app.on_event("startup")
async def startup_event():
    # fazemos set do tracking URI para apontar para o ficheiro de db
    # onde estão os metadados dos nossos modelos registrados
    with open(r'C:\Users\polin\Downloads\projecto_final_OML\projecto_final\rumos_bank\config\app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(config["tracking_uri"])


# endpoint de predict que será chamado para receber pedidos com os inputs para o modelo
# e que vai retornar na resposta a previsão do modelo
@app.post("/predict")
async def root(input: Request):  
    # lê a config da app
    with open(r'C:\Users\polin\Downloads\projecto_final_OML\projecto_final\rumos_bank\config\app.json') as f:
        config = json.load(f)
    # fazemos load do modelo registado
    # de acordo com o model name e model version lidos da config
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{config['model_name']}/{config['model_version']}"
    )
    # construimos um dataframe com os inputs do modelo que recebmos no pedido
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.dict().items()})
    # chamamos a função de predict do modelo e temos a previsao do mesmo
    prediction = model.predict(input_df)
    # retornamos como resposta um diccionário com a predição associada à chave "prediction"
    return {"prediction": prediction.tolist()[0]}


uvicorn.run(app=app, port=5006)
