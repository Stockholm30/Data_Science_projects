import os
import settings
import pandas as pd


HEADERS={
    "ga_hits": [
        'session_id', 'hit_date', 'hit_time', 'hit_number', 'hit_type',
       'hit_referer', 'hit_page_path', 'event_category', 'event_action',
       'event_label', 'event_value'
    ],
  "ga_sessions": [
        'session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
       'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
       'utm_keyword', 'device_category', 'device_os', 'device_brand',
       'device_model', 'device_screen_resolution', 'device_browser',
       'geo_country', 'geo_city'
    ]
}

SELECT = {
    "ga_hits": ['session_id', 'hit_date', 'hit_time', 'hit_type',
               'hit_page_path', 'event_category', 'event_action'],
    "ga_sessions": [
         'session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
       'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
      'device_category', 'device_os', 'device_brand',
       'device_screen_resolution', 'device_browser',
       'geo_country', 'geo_city'
    ]
}


def concatenate(prefix="ga_hits"):
    files = os.listdir(settings.DATA_DIR)
    full = []
    for f in files:
        if not f.startswith(prefix):
            continue

        data = pd.read_csv(os.path.join(settings.DATA_DIR, f), sep=",", header=None, names=HEADERS[prefix], index_col=False)
        data = data[SELECT[prefix]]
        full.append(data)

    full = pd.concat(full, axis=0)

    full.to_csv(os.path.join(settings.PROCESSED_DIR, "{}.csv".format(prefix)), sep=",", header=SELECT[prefix], index=False)

if __name__ == "__main__":
    concatenate("ga_hits")
    concatenate("ga_sessions")
