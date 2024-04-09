import dill
import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import os
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer


prepared_data = pd.read_csv('processed/prepared_data')
sample_prepared = prepared_data.sample(frac=0.05)
sample_prepared = sample_prepared.drop(['id'], axis=1)
features = sample_prepared.columns.tolist()[:-1]
target = ['flag']
X= sample_prepared[features]
y = sample_prepared[target]

over = RandomOverSampler(sampling_strategy=0.8)
 X_resampled, y_resampled = over.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_test.to_csv('processed/X_test_examples')
numeric_features = ['pre_fterm',
                                        'pre_till_fclose',
                                        'pre_loans_credit_cost_rate', 'pre_util', 'enc_paym_5', 'enc_paym_6',
                                        'enc_paym_7', 'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11',
                                        'enc_paym_12', 'enc_paym_13', 'enc_paym_14']
numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

preprocessor = ColumnTransformer(transformers=[
            ("num_transform", numeric_transformer, numeric_features)
        ])
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier())])
model =  pipe.fit(X_train, y_train)
predicted_target = model.predict(X_test)
score = roc_auc_score(predicted_target,y_test)
param_grid = [
            {
                "classifier__min_samples_leaf":  [2, 4, 10, 15],
                "classifier__max_depth" : [2, 5, 15, 20],
                'classifier__max_leaf_nodes': [100, 200, 300, 500, 10000]

            }
        ]

grid_search = GridSearchCV(pipe, param_grid, scoring='roc_auc',  cv=10, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
if grid_search.best_score_ > score:
            best_score = grid_search.best_score_
            new_model = DecisionTreeClassifier(max_depth=20,max_leaf_nodes= 10000, min_samples_leaf= 2)
            pipe = Pipeline(steps=[("preprocesser", preprocessor), ("classifier", new_model)])
            model = pipe.fit(X_train, y_train)

else:
            pipe= Pipeline(steps=[("preprocesser", preprocessor), ("classifier", DecisionTreeClassifier())])
            model = pipe.fit(X_train, y_train)
            best_score = score

with open('credit_risk_pipe.pkl', 'wb') as file:
            dill.dump({
                'model': pipe,
                'metadata': {
                    "name": "Credit Risk model",
                    "author": "Sofya Trifonova",
                    "version": 1,

                    "type": 'DecisionTreeClassifier',
                    "accuracy": best_score
                }
            }, file)
