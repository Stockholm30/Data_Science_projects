import pandas as pd
import settings 
from datetime import datetime


data_hits_messy = pd.read_csv(settings.PROCESSED_DIR+'/'+'ga_hits.csv')[1:]
data_sessions_messy = pd.read_csv(settings.PROCESSED_DIR+'/'+'ga_sessions.csv')[1:]

#DATA_HITS (Nan-cleaning)
#it is problematic to determine the time, since only the time of the first entry to the site is known, 
#but the time when the user stops activity on the site is not fixed, 
#so it is reasonable to replace NaN with an average value for event_category
data_timena = data_hits_messy[data_hits_messy['hit_time'].isna()]
data_timenotna = data_hits_messy[data_hits_messy['hit_time'].notna()]
empty_cat_times = list(set(data_timena.event_category.tolist()))

#grouping the average time by event category
agg_func_math = {
    'hit_time': ['mean']
}
mean_times_cat = data_timenotna.groupby(['event_category']).agg(agg_func_math)
mean_times_cat =mean_times_cat.reset_index()
cat_empty = mean_times_cat[('event_category',     '')].tolist()
#we look at which categories are NaN in time, if there are categories for which there is no average time for a dataset with a known time, 
#then we assign it the average time for all categories of the mean_all action
for i in range(len(empty_cat_times)):
    cat = empty_cat_times[i]
    mean_all = data_hits.hit_time.describe().mean()
    if cat in cat_empty:
        time = mean_times_cat[mean_times_cat[('event_category',     '')].isin([cat])]['hit_time']['mean'].tolist()[0]
        data_hits_messy.loc[(data_hits_messy['hit_time'].isna()&data_hits_messy['event_category'].isin([cat])),('hit_time')] = time
    else:
        data_hits_messy.loc[(data_hits_messy['hit_time'].isna()&data_hits_messy['event_category'].isin([cat])),('hit_time')] = mean_all
  
#DATA HITS (Abnormal values and types)
data_hits_messy.hit_time = data_hits_messy.hit_time.astype(int)
#convert to seconds for convenience
data_hits_messy['hit_time'] =(data_hits_messy['hit_time']/1000).round()

#from hit_date, we will separate the day of the week, as well as separate the month and year
dates = data_hits_messy.hit_date.tolist()
days= []
for i in range(len(dates)):
    date = dates[i]
    day = datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]))
    if datetime.weekday(day) == 0:
        days.append('monday')
    elif(datetime.weekday(day) == 1):
        days.append('tuesday')
    elif(datetime.weekday(day) == 2):
        days.append('wednesday')
    elif(datetime.weekday(day) == 3):
        days.append('thursday')
    elif(datetime.weekday(day) == 4):
        days.append('friday')
    elif(datetime.weekday(day) == 5):
        days.append('saturday')
    elif(datetime.weekday(day) == 6):
        days.append('sunday')
data_hits['day_of_week'] = days

month= []
for i in range(len(dates)):
    date = dates[i]
    day = datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]))
    month.append(day.month)
data_hits_messy['month'] = month

years= []
for i in range(len(dates)):
    date = dates[i]
    day = datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]))
    years.append(day.year)
data_hits_messy['year'] = years

#let's choose car brands and create a new feature
sessions = list(set(data_hits_messy.session_id.tolist()))
urls= list((data_hits_messy.hit_page_path.tolist()))
auto_marks = []
for i in range(len(urls)):
    if 'all/' in urls[i]:
        mark = urls[i].split('/')[3]
        auto_marks.append(mark)
    else:
        auto_marks.append('other')
data_hits_messy['auto_mark']=auto_marks
#the purposes are related to whether the client will subscribe
purposes = ['sub_car_claim_click', 'sub_car_claim_submit_click',
'sub_open_dialog_click', 'sub_custom_question_submit_click',
'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
'sub_car_request_submit_click']
actions = data_hits_messy.event_action.tolist()

new_actions=[]
for i in range(len(actions)):
    if actions[i] in purposes:
        new_actions.append('1')
    else:
        new_actions.append('0')
data_hits_messy['purpose_action'] = new_actions 
#new feature
data_hits_messy['len_link']=data_hits_messy['hit_page_path'].apply(len)
data_hits_messy.to_csv(os.path.join(settings.PROCESSED_DIR2, "data_hits_clean.csv",  index=False)

