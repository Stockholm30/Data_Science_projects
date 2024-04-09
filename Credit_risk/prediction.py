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


targets = pd.read_csv('/train_target.csv')
        prepared_data  = data_all.merge(targets, left_on='id', right_on='id')
        sample_prepared = prepared_data.sample(frac=0.05)
        sample_prepared = sample_prepared.drop(['id'], axis=1)
        features = sample_prepared.columns.tolist()[:-1]
        target = ['flag']
        X= sample_prepared[features]
        y = sample_prepared[target]

        over = RandomOverSampler(sampling_strategy=0.8)
        X_resampled, y_resampled = over.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        X_test_scaled.to_csv('X_test_examples')


        model = DecisionTreeClassifier()
        pipe = Pipeline(steps= [('classifier', model)])

        model=  pipe.fit(X_train_scaled, y_train)
        best_score = roc_auc_score( model.predict(X_test_scaled),y_test )
        print(best_score)

        pipe.fit(X, y)
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
