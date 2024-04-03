import pandas as pd

def train_composition(data_1, data_2):
  data_1 = data_1.drop_duplicates()
  sessions = list(set(data_1.session_id.tolist()))
  known_sessions_data = data_2[data_2['session_id'].isin(sessions)]
  train_data = known_sessions_data.merge(data_1, on=["session_id"])
  #Based on EDA(see in project EDA_sberauto)
  train_data2 = train_data[['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
       'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
       'device_category', 'device_os', 'device_brand',
       'device_screen_resolution', 'device_browser', 'geo_country',
       'geo_city', 'purpose_action']]
  train_data = train_data2.drop_duplicates()
  def get_month_1(x):
    return (x.month)
  train_data['month']=train_data['visit_date'].apply(get_month_1)
  months = [5, 6, 7, 8, 9, 10, 11, 12]
  features_important=['utm_source', 'utm_campaign', 'utm_adcontent', 'device_screen_resolution', 'device_brand' , 'geo_city',
                   'geo_country']
  features= [
       'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'geo_city',
       'device_category', 'device_os', 'device_brand',
       'device_screen_resolution', 'device_browser', 'geo_country']
  important =[]
  not_important = []
  new_data = pd.DataFrame()
  for i in range(len(months)):

    if months[i] == 5:
        month_clients = list(set(train_data[train_data['month'].isin([months[i]])].client_id.tolist()))
        month_data = train_data[train_data['month'].isin([months[i]])]
        client_data = month_data[month_data['client_id'].isin(month_clients)]
        
       
        for l in range(len(features_important)):
            important_ = client_data[features_important[l]].value_counts().index[:30].tolist()
            n_important_ = list(set(client_data[features_important[l]].tolist())-set(important_))
            important_data = client_data[client_data[features_important[l]].isin(important_)]
            n_important_data = client_data[client_data[features_important[l]].isin(n_important_)]
            n_important_data[features_important[l]]='other'
            client_data_new = pd.concat([n_important_data, important_data])
            client_data_new['client_id']='new'
            client_data =client_data_new
        new_data = pd.concat([new_data, client_data])      
    elif(months[i] > 5):
        month_clients = list((set(train_data[train_data['month'].isin([months[i]])].client_id.tolist()))-
                             (set(train_data[train_data['month'].isin([months[i]-1])].client_id.tolist())))
        old_clients = train_data[train_data['month'].isin([months[i]-1])].client_id.tolist()
        month_data2 = train_data[train_data['month'].isin([months[i]])]
        old_client_data = month_data2[month_data2['client_id'].isin(old_clients)]
        
        month_data = train_data[train_data['month'].isin([months[i]])]
        client_data = month_data[month_data['client_id'].isin(month_clients)]
        for l in range(len(features_important)):
            important_ = client_data[features_important[l]].value_counts().index[:30].tolist()
        #print(important_)
            n_important_ = list(set(client_data[features_important[l]].tolist())-set(important_))
        #print(len(n_important_))
            important_data = client_data[client_data[features_important[l]].isin(important_)]
            n_important_data = client_data[client_data[features_important[l]].isin(n_important_)]
            n_important_data[features_important[l]]='other'
            client_data_new = pd.concat([n_important_data, important_data])
            imp_old= old_client_data[old_client_data[features_important[l]].isin(important_)]
            not_imp_old = old_client_data[old_client_data[features_important[l]].isin(n_important_)]
            not_imp_old[features_important[l]]='other'
            client_data_old_new = pd.concat([imp_old, not_imp_old])
            client_data_old_new['client_id']='old'
            old_client_data = client_data_old_new
            client_data_new['client_id']='new'
            client_data =client_data_new
        data_all = pd.concat([client_data_new, client_data_old_new])
        new_data = pd.concat([new_data, data_all])
    model_data = pd.DataFrame()
    for i in range(len(months)):
      month_data = new_data[new_data['month'].isin([months[i]])]
      shuffled_df = month_data.sample(frac=1,random_state=4)

      purpose_df = shuffled_df.loc[shuffled_df['client_id'] == 'old']
      non_purpose_df = shuffled_df.loc[shuffled_df['client_id'] == 'new'].sample(n=len(purpose_df),random_state=42)

      model_data2 = pd.concat([purpose_df, non_purpose_df])
      model_data = pd.concat([model_data, model_data2])
    old = new_data[new_data['month'].isin([12])]
    only_new = new_data[new_data['month'].isin([5])]
    only_old =old[old['client_id'].isin(['old'])]
    new_=pd.concat([only_new, only_old])
    shuffled_df = new_.sample(frac=1,random_state=4)
    purpose_df = shuffled_df.loc[shuffled_df['client_id'] == 'old']
    non_purpose_df = shuffled_df.loc[shuffled_df['client_id'] == 'new'].sample(n=len(purpose_df),random_state=42)
    model_data_= pd.concat([purpose_df, non_purpose_df])
    model_data_total = pd.concat([model_data, model_data_])
    shuffled_df = model_data_total.sample(frac=1,random_state=4)
    purpose_df = shuffled_df.loc[shuffled_df['purpose_action'] == 1]
    non_purpose_df = shuffled_df.loc[shuffled_df['purpose_action'] == 0].sample(n=len(purpose_df),random_state=42)
    model_data_balanced = pd.concat([purpose_df, non_purpose_df])
    columns_to_drop =model_data_balanced.columns[0:5]
    columns_to_drop[0:5]
    model_data_balanced_ = model_data_balanced.drop(columns_to_drop, axis=1)
    model_data_balanced_.pop('month')
    model_data_balanced_.to_csv(settings.PROCESSED_DIR2+'/'+ "sberauto_train_data.csv",  index=False)
                                                    

def read():
    data_sessions = pd.read_csv(settings.PROCESSED_DIR2+'/'+'data_sessions_clean.csv')
    data_hits = pd.read_csv(settings.PROCESSED_DIR2+'/'+'data_hits_clean.csv')
    return data_hits, data_sessions

if __name__ == "__main__":
    data = read()
    train_result = annotate(data[0], data[1])
  




