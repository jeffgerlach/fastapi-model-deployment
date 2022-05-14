from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the FastAPI Census Income Data Set Model API!"}


# 0 result
less_than_50_input = {
    "age": 70,
    "workclass": "Private",
    "fnlgt": 195739,
    "education": "10th",
    "education-num": 6,
    "marital-status": "Widowed",
    "occupation": "Craft-repair",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States"
}

# 1 result
greater_than_50_input = {
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


def test_0_inference():
    r = client.post("/predict", json=less_than_50_input)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_1_inference():
    r = client.post("/predict", json=greater_than_50_input)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
