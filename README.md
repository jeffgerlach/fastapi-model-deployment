# Deploying a Machine Learning Model on Heroku with FastAPI

This is the final project in the [Udacity Machine Learning DevOps Engineer NanoDegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).
The code was forked from [the course repository](https://github.com/udacity/nd0821-c3-starter-code).

## Overview
A [random forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) was trained on the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income) 
from the UCI Machine Learning Repository to predict whether a person makes more than $50,000/year based on their census response.
Information on the model can be found in the [model card](https://github.com/jeffgerlach/fastapi-model-deployment/blob/master/starter/model_card.md).

The input data and exported model artifacts were tracked using [DVC](https://dvc.org/) with the artifacts store on Amazon S3. A CI/CD pipeline was set up using GitHub actions to lint the code
using `flake8` and run tests with `pytest`.

A [FastAPI](https://fastapi.tiangolo.com/) API was developed to allow users to POST input data to the model and receive a prediction from the model. The [API documentation](https://udacity-fastapi-model.herokuapp.com/docs) provides examples on how to use the API.

If the CI/CD pipeline passes, the model is the deployed on [Heroku](https://github.com/jeffgerlach/fastapi-model-deployment) where users can interact with it (after the dyno spins up if it was idle).

## Basic usage:

[API documentation](https://udacity-fastapi-model.herokuapp.com/docs)

Get a welcome message:

  Using `curl`: 
```bash
curl -X 'GET' \
  'https://udacity-fastapi-model.herokuapp.com/' \
  -H 'accept: application/json'
 ```
Response:
```json
{
  "greeting": "Welcome to the FastAPI Census Income Data Set Model API!"
}
```

Get a salary prediction from the model:

  Using `curl`: 
```bash
curl -X 'POST' \
  'https://udacity-fastapi-model.herokuapp.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 25,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 2174,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}'
```
Response:
```json
{
  "prediction": "<=50K"
}
```


## Project links:
* [Github repository](https://github.com/jeffgerlach/fastapi-model-deployment)
* [FastAPI GET URL](https://udacity-fastapi-model.herokuapp.com/)
* [FastAPI POST URL](https://udacity-fastapi-model.herokuapp.com/predict)
