# FastAPI code
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference
from starter.starter.train_model import cat_features

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


def replace_hyphen(string: str) -> str:
    return string.replace('-', '_')


class InferenceInput(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

    class Config:
        alias_generator = replace_hyphen


class InferenceOutput(BaseModel):
    prediction: str = Field(..., example=">50K")


encoder = joblib.load('starter/model/random_forest_onehot_encoder')
rf_model = joblib.load('starter/model/trained_random_forest')
lb = joblib.load('starter/model/label_binarizer')


@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the FastAPI Census Income Data Set Model API!"}


@app.post("/predict", response_model=InferenceOutput)
async def make_inference(input_data: InferenceInput):
    df = pd.DataFrame.from_dict([input_data.dict(by_alias=True)])
    X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder)
    predictions = lb.inverse_transform(inference(rf_model, X))
    output_value = str(predictions[0])

    return {"prediction": output_value}

if __name__ == "__main__":
    uvicorn.run(app)
