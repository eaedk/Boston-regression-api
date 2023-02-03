# Imports
from fastapi import FastAPI
import pickle, uvicorn, os
from pydantic import BaseModel
import pandas as pd
import numpy as np

####################################################################
# Config & Setup
## Variables of environment
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, "assets")
ml_components_pkl = os.path.join(ASSETSDIRPATH, "ml-components.pkl")

print(
    f" {'*'*10} Config {'*'*10}\n INFO: DIRPATH = {DIRPATH} \n INFO: ASSETSDIRPATH = {ASSETSDIRPATH} "
)

## API Basic Config
app = FastAPI(
    title="House Pricing API",
    version="0.0.1",
    description="Prediction of Boston house pricing",
)


## Loading of assets
with open(ml_components_pkl, "rb") as f:
    loaded_items = pickle.load(f)
print("INFO:    Loaded assets:", loaded_items)

ml_model = loaded_items["model"]
scaler = loaded_items["scaler"]

####################################################################
# API Core
## BaseModel
class ModelInput(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float


## Utils
def processing_FE(
    dataset, scaler, imputer=None, encoder=None, FE=None
):  # FE : ColumnTransfromer, Pipeline
    "Cleaning, Processing and Feature Engineering of the input dataset."
    """:dataset pandas.DataFrame"""

    if imputer is not None:
        output_dataset = imputer.transform(dataset)
    else:
        output_dataset = dataset.copy()

    output_dataset = scaler.transform(output_dataset)

    if encoder is not None:
        output_dataset = encoder.transform(output_dataset)
    if FE is not None:
        output_dataset = FE.transform(output_dataset)

    return output_dataset


def make_prediction(
    CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
):
    """"""
    df = pd.DataFrame(
        [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
        columns=[
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
        ],
    )
    # Dataframe visualization
    print("*"*30,"Input Dataframe","*"*30)
    print(df.to_string(justify="center")) 
    print("*"*70)
    print("*"*30,"Input Dataframe Info","*"*30)
    print(df.info())
    print("*"*70)

    X = processing_FE(dataset=df, scaler=scaler, imputer=None, encoder=None, FE=None)

    model_output = ml_model.predict(X).tolist()
    print("INFO:    PREDICTION DONE")

    # print(type(model_output))
    # print(model_output)

    return model_output


## Endpoints
@app.post("/boston")
async def predict(input: ModelInput):
    """__descr__

    --details---
    """
    output_pred = make_prediction(
        CRIM=input.CRIM,
        ZN=input.ZN,
        INDUS=input.INDUS,
        CHAS=input.CHAS,
        NOX=input.NOX,
        RM=input.RM,
        AGE=input.AGE,
        DIS=input.DIS,
        RAD=input.RAD,
        TAX=input.TAX,
        PTRATIO=input.PTRATIO,
        B=input.B,
        LSTAT=input.LSTAT,
    )
    return {
        "prediction": output_pred,
        "input": input,
    }


####################################################################
# Execution

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        reload=True,
    )
