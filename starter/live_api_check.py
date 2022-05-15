import requests
import json

input = {
    "age": 43,
    "workclass": "Private",
    "fnlgt": 339814,
    "education": "Some-college",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5178,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

res = requests.post("https://udacity-fastapi-model.herokuapp.com/predict", data=json.dumps(input))
print("Status code:", res.status_code)
print(res.json())
