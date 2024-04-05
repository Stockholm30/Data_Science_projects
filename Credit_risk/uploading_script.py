path = 'train_data/' 

import os
import pandas as pd
import tqdm

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
                                                                 , num_parts_to_preprocess_at_once, columns=['id','pre_since_opened',
       'is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060',
       'is_zero_loans6090', 'is_zero_loans90', 'is_zero_util',
       'is_zero_over2limit', 'is_zero_maxover2limit', 'pclose_flag',
       'fclose_flag',  'enc_paym_6',
       'enc_paym_7', 'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11',
       'enc_paym_12', 'enc_paym_13', 'enc_paym_14'],
                                                                 verbose=verbose)

          # data preprocessing

                agg_func_count_application = {
                              'pre_since_opened':'mean',
                              'is_zero_loans5':'max',
                              'is_zero_loans530':'max',
                              'is_zero_loans3060':'max',
                              'is_zero_loans6090':'max',
                              'is_zero_loans90':'max',
                               'is_zero_util':'max',
                              'is_zero_over2limit':'max',
                               'is_zero_maxover2limit':'max',
                              'pclose_flag':'max',
                               'fclose_flag':'max',
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
                reduced_data["enc_paym_6"] = reduced_data['enc_paym_6'].astype('int')
                reduced_data["enc_paym_7"] = reduced_data['enc_paym_7'].astype('int')
                reduced_data["enc_paym_8"] = reduced_data['enc_paym_8'].astype('int')
                reduced_data["enc_paym_9"] = reduced_data['enc_paym_9'].astype('int')
                reduced_data["enc_paym_10"] = reduced_data['enc_paym_10'].astype('int')
                reduced_data["enc_paym_11"] = reduced_data['enc_paym_11'].astype('int')
                reduced_data["enc_paym_12"] = reduced_data['enc_paym_12'].astype('int')
                reduced_data["enc_paym_13"] = reduced_data['enc_paym_13'].astype('int')
                reduced_data["enc_paym_14"] = reduced_data['enc_paym_14'].astype('int')


                def calculate_outliers(data):
                    q25 = data.quantile(0.25)
                    q75 = data.quantile(0.75)
                    iqr = q75 - q25

                    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
                    return boundaries

                boundaries_pre_since_opened = calculate_outliers(reduced_data['pre_since_opened'])
                is_outlier_pre_since_opened1 = (reduced_data.pre_since_opened < boundaries_pre_since_opened[0])
                is_outlier_pre_since_opened2 = (reduced_data.pre_since_opened > boundaries_pre_since_opened[1])

                reduced_data.loc[is_outlier_pre_since_opened1, 'pre_since_opened'] = int(boundaries_pre_since_opened[0])
                reduced_data.loc[is_outlier_pre_since_opened2, 'pre_since_opened'] = int(boundaries_pre_since_opened[1])




                # writing the prepared data to a file
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
data_all.to_csv('processed/data_test')
       
