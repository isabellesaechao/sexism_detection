import csv
import numpy as np
import pandas as pd
import xgboost
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import f1_score


binary_data = "src/data/preprocessed_data_aggregated.csv"
fourway_data = "src/data/preprocessed_4way_data_aggregated.csv"
elevenway_data = "src/data/preprocessed_11way_data_aggregated.csv"


"""
Task A: Classify based on sexist and non-sexist labels
"""
# preprocess the 'text' using bag of words
df = pd.read_csv(binary_data)
train_data = df[df['split'] == 'train']
test_data = df[df['split'] == 'test']
dev_data = df[df['split'] == 'dev']

vectorizer = TfidfVectorizer(stop_words='english')

# map binary labels to 0 and 1
le = LabelEncoder()
label_map = {0: 'not sexist', 1: 'sexist'}


# Use the prepocessed csvs files with text as X and label_sexist as y

X_train, y_train =  vectorizer.fit_transform(train_data['cleaned_text'].astype(str).values), train_data['label'].values
X_dev, y_dev = vectorizer.transform(dev_data['cleaned_text'].astype(str).values), dev_data['label'].values
X_test, y_test = vectorizer.transform(test_data['cleaned_text'].astype(str).values), test_data['label'].values

dtrain = xgboost.DMatrix(X_train, label=y_train)
ddev = xgboost.DMatrix(X_dev, label=y_dev)
dtest = xgboost.DMatrix(X_test, label=y_test)

# Training the xgboost model
params = {
    'objective': 'binary:logistic',
    'max_depth': 10,
    'eta': 0.1,
    'eval_metric': 'error',
    'gamma': 0.5
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
}

model = xgboost.train(params, dtrain, evals=[(dtrain, 'train')], num_boost_round=100, verbose_eval=10, early_stopping_rounds=10)
init_dev_predictions = model.predict(ddev)

# Calculate scores for Task A
y_pred_dev = [1 if p > 0.5 else 0 for p in init_dev_predictions]
f1 = f1_score(y_dev, y_pred_dev)
print("Task A dev results:")
print(f'F1 score: {f1}')
print()

"""
Task B: For those labelled sexist, classify on the four:
"""
# For Task B and C only use data that is sexist
df = pd.read_csv(fourway_data)
train_data = df[df['split'] == 'train']
test_data = df[df['split'] == 'test']
dev_data = df[df['split'] == 'dev']


# Class labels for sexism categories
label_map2 = {
    0: '1. threats, plans to harm and incitement',
    1: '2. derogation',
    2: '3. animosity',
    3: '4. prejudiced discussions'
    }

X_train, y_train =  vectorizer.fit_transform(train_data['cleaned_text'].astype(str).values), train_data['label'].values
X_dev, y_dev = vectorizer.transform(dev_data['cleaned_text'].astype(str).values), dev_data['label'].values
X_test, y_test = vectorizer.transform(test_data['cleaned_text'].astype(str).values), test_data['label'].values

dtrain = xgboost.DMatrix(X_train, label=y_train)
ddev = xgboost.DMatrix(X_dev, label=y_dev)
dtest = xgboost.DMatrix(X_test, label=y_test)


# Training the xgboost model
params = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'max_depth': 10,
    'eta': 0.1,
    'eval_metric': 'merror'
    # 'gamma': 0.5,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8
}

model_sexism_cat = xgboost.train(params, dtrain, evals=[(dtrain, 'train')], num_boost_round=100, verbose_eval=10, early_stopping_rounds=10)
init_dev_predictions_sexism_cat = model_sexism_cat.predict(ddev)

# Calculate scores for Task B
f1 = f1_score(y_dev, init_dev_predictions_sexism_cat, average='macro')
print("Task B dev results:")
print(f'F1 score: {f1}')
print("---------")



"""
Task C: For those labelled sexist and in a specific category classify on the eleven:
"""
# using the eleven way preprocessing
df = pd.read_csv(elevenway_data)
train_data = df[df['split'] == 'train']
test_data = df[df['split'] == 'test']
dev_data = df[df['split'] == 'dev']

# Class labels for sexism subcategories
label_map3 = {
    0: '1.1 threats of harm',
    1: '1.2 incitement and encouragement of harm',
    2: '2.1 descriptive attacks',
    3: '2.2 aggressive and emotive attacks',
    4: '2.3 dehumanising attacks & overt sexual objectification',
    5: '3.1 casual use of gendered slurs, profanities, and insults',
    6: '3.2 immutable gender differences and gender stereotypes',
    7: '3.3 backhanded gendered compliments',
    8: '3.4 condescending explanations or unwelcome advice',
    9:'4.1 supporting mistreatment of individual women',
    10: '4.2 supporting systemic discrimination against women as a group'
    }

X_train, y_train =  vectorizer.fit_transform(train_data['cleaned_text'].astype(str).values), train_data['label'].values
X_dev, y_dev = vectorizer.transform(dev_data['cleaned_text'].astype(str).values), dev_data['label'].values
X_test, y_test = vectorizer.transform(test_data['cleaned_text'].astype(str).values), test_data['label'].values

dtrain = xgboost.DMatrix(X_train, label=y_train)
ddev = xgboost.DMatrix(X_dev, label=y_dev)
dtest = xgboost.DMatrix(X_test, label=y_test)


# Training the xgboost model
params = {
    'objective': 'multi:softmax',
    'num_class': 11,
    'max_depth': 10,
    'eta': 0.1,
    'eval_metric': 'merror'
    # 'gamma': 0.5,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8
}

model_sexism_subcat = xgboost.train(params, dtrain, evals=[(dtrain, 'train')], num_boost_round=100, verbose_eval=10, early_stopping_rounds=10)
init_dev_predictions_sexism_subcat = model_sexism_subcat.predict(ddev)


# Calculate scores for Task B
f1 = f1_score(y_dev, init_dev_predictions_sexism_subcat, average='macro')
print("Task C dev results:")
print(f'F1 score: {f1}')
print("---------")
