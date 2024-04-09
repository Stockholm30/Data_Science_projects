path = 'train_data/' 

import os
import pandas as pd
import tqdm

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

                #for i in range(len(features_type_change)):
                 #   reduced_data[features_type_change[i]] = reduced_data[features_type_change[i]].astype('int')

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

targets = pd.read_csv('target_data/train_target.csv')
data_all = prepare_transactions_dataset(path, num_parts_to_preprocess_at_once=1, num_parts_total=12,
                                               save_to_path='train_data_/')
prepared_data  = data_all.merge(targets, left_on='id', right_on='id')
prepared_data.to_csv('processed/prepared_data')
       
