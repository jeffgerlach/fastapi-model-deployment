# Model Card

## Model Details

This model was developed by Jeff Gerlach in May 2022 as a final project for
the [Udacity Machine Learning DevOps Engineer NanoDegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)
.
The model selected was
a [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
from ScikitlLearn v1.0.2. Default hyper-parameters were used for the initial version.

## Intended Use

The intent of this model is to predict whether an adult's yearly salary is either over or under/equal to $50,000 based
on public US Census data.

## Training Data

The model was trained using a provided subset of
the [Census Income Dataset available from the UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income)
.
Data cleaning involved removing all spaces from the existing CSV file. There are 32,561 entries in the cleaned CSV file
with a 80% train split. The target label for prediction is the 'salary'
feature which can be either `>50K` or `<=50K`. Data pre-processing includes one hot encoding for the categorical
features and a label binarizer is used on the target label.

The following list shows all of the features available in this data set:

```
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
```

Citation: `Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.`

## Evaluation Data

The model is evaluated using a 20% test split from the original cleaned data used to train the model.

## Metrics

Metrics used for model evaluation include:

- precision
- recall
- F<sub>β</sub>

Results:

| metric        | value |
|---------------|-------|
| precision     | 0.75  |
| recall        | 0.621 |
| F<sub>β</sub> | 0.669 |

## Ethical Considerations
The model was trained on US census data, so bias could be introduced based on the demographic composition
of the US in 1994. The data used was publicly available census responses. Explicit bias checks were not performed.

## Caveats and Recommendations
The training data is based on information collected over 25 years ago, so it may not be as accurate if used with 
present day inputs (due to shifting demographics and inflation). More recent survey data would be likely be more
accurate for predicting salaries in the present. The target labels in the dataset are also skewed towards lower
salaries, with the following distribution:

| label   | % of total |
|---------|------------|
| "<=50K" | 75.9%      |
| ">50K"  | 24.1%      |
