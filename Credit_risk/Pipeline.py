from datetime import datetime

import dill
import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import tqdm
import os


from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV

def main():

        print('Pipeline Credit Risk')
        path = 'train_data_/'
        def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                        num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:

            res = []
            dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                                if filename.startswith('train')])
            print(dataset_paths)

            start_from = max(0, start_from)
            chunks = dataset_paths[start_from: start_from + num_parts_to_read]
            if verbose:
                print('Reading chunks:\n')
                for chunk in chunks:
                    print(chunk)
            for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
                print('chunk_path', chunk_path)
                chunk = pd.read_parquet(chunk_path, columns=columns)
                res.append(chunk)

            return pd.concat(res).reset_index(drop=True)

        def prepare_transactions_dataset(path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1,
                                     num_parts_total: int = 50,
                                     save_to_path=None, verbose: bool = False):

            preprocessed_frames = []

            for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                                       desc="Transforming transactions data"):
                transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step
                                                                 , num_parts_to_preprocess_at_once, columns=
                                                                     None , verbose=verbose)

            # здесь должен быть препроцессинг данных

                agg_func_count_application = {
                              'pre_fterm':'mean',
                              'pre_till_fclose':'max',
                              'pre_loans_credit_cost_rate':'median',
                              'pre_util':'max',
                               'enc_paym_5':'max',
                              'enc_paym_6':'median',
                              'enc_paym_7':'median',
                            'enc_paym_8':'median',
                              'enc_paym_9':'median',
                              'enc_paym_10':'median',
                              'enc_paym_11':'median',
                              'enc_paym_12':'median',
                          'enc_paym_13':'median',
                              'enc_paym_14':'median'

                              }

                reduced_data = transactions_frame.groupby(['id']).agg(agg_func_count_application)

                def calculate_outliers(data):
                    q25 = data.quantile(0.25)
                    q75 = data.quantile(0.75)
                    iqr = q75 - q25

                    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
                    return boundaries

                features_type_change = ['pre_fterm',
                                        'pre_till_fclose',
                                        'pre_loans_credit_cost_rate', 'pre_util', 'enc_paym_5', 'enc_paym_6',
                                        'enc_paym_7', 'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11',
                                        'enc_paym_12', 'enc_paym_13', 'enc_paym_14']

                for i in range(len(features_type_change)):
                    boundaries = calculate_outliers(reduced_data[features_type_change[i]])
                    is_outlier_1 = (reduced_data[features_type_change[i]] < boundaries[0])
                    is_outlier_2 = (reduced_data[features_type_change[i]] > boundaries[1])
                    if len(is_outlier_1) == 0:
                        reduced_data.loc[is_outlier_2, features_type_change[i]] = int(boundaries[1])
                    elif(len(is_outlier_2) == 0):
                        reduced_data.loc[is_outlier_1, features_type_change[i]] = int(boundaries[0])
                    else:
                        reduced_data.loc[is_outlier_1, features_type_change[i]] = int(boundaries[0])
                        reduced_data.loc[is_outlier_2, features_type_change[i]] = int(boundaries[1])

                for i in range(len(features_type_change)):
                    reduced_data[features_type_change[i]] = reduced_data[features_type_change[i]].astype('int')

                # записываем подготовленные данные в файл
                if save_to_path:
                    block_as_str = str(step)
                    if len(block_as_str) == 1:
                        block_as_str = '00' + block_as_str
                    else:
                        block_as_str = '0' + block_as_str
                    reduced_data.to_parquet(os.path.join(save_to_path, f'processed_chunk_{block_as_str}.parquet'))

                preprocessed_frames.append(reduced_data)
            return pd.concat(preprocessed_frames)


        data_all = prepare_transactions_dataset(path, num_parts_to_preprocess_at_once=1, num_parts_total=12,
                                               save_to_path='train_data_/')
        #data_all.to_csv('data_test')
        #data = pd.read_csv('data_test')

        targets = pd.read_csv('data/train_target.csv')
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

        #scaler = StandardScaler()
        #X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        X_test.to_csv('X_test_examples')
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
        pipe = Pipeline(steps=[("preprocesser", preprocessor), ("classifier", DecisionTreeClassifier())])
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




if __name__ == '__main__':
    main()

