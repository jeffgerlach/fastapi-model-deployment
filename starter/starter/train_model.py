# Script to train machine learning model.
import logging
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, inference, \
    compute_model_metrics, \
    compute_model_slice_metrics

# Add the necessary imports for the starter code.
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
logger.info("Reading data")
data = pd.read_csv("../data/census_cleaned.csv")

logger.info("Splitting data")
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the test data with the process_data function.
logger.info("Processing data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
logger.info("Training model")
rf_model = train_model(X_train, y_train)

logger.info("Testing model")
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder,
    lb=lb)
test_predictions = inference(rf_model, X_test)

logger.info("Computing test metrics")
precision, recall, f_beta = compute_model_metrics(y_test, test_predictions)
logger.info("=" * 60)
logger.info(
    f"Model test metrics: precision: {precision:.3f} recall: {recall:.3f} "
    f"f_beta: {f_beta:.3f}")
logger.info("=" * 60)

logger.info("Computing model test metrics on all categorical feature slices:")
slice_metrics = {}
# run slices on all options for all categorical features and put into a dict
# to be output to a file
for cat_feature in cat_features:
    vertical_slice = test[cat_feature]
    categorical_metrics = compute_model_slice_metrics(rf_model, X_test, y_test,
                                                      vertical_slice)
    logger.info(categorical_metrics)
    slice_metrics[cat_feature] = categorical_metrics

logger.info("Saving model")
joblib.dump(rf_model, "../model/trained_random_forest")

logger.info("Saving encoder")
joblib.dump(encoder, "../model/random_forest_onehot_encoder")

logger.info("Saving metrics")
with open("../model/slice_output.txt", "w", encoding="utf8") as json_file:
    json.dump(slice_metrics, json_file)
