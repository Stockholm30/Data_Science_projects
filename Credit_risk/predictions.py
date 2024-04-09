import pandas as pd

import joblib

def main():
    model = joblib.load('model/credit_risk_pipe.pkl')
    data_test = pd.read_csv('model/X_test_examples')
    data_test.pop('Unnamed: 0')
    test_res=[]
   # data_test = data_test[0:10]
    for i in range(len(data_test)):
        y = model['model'].predict(pd.DataFrame(data_test.iloc[i]).T)
        test_res.append(['Result is:', y[0]])
    test_res_df= pd.DataFrame(test_res, columns=['num', 'result'])
    test_res_df.to_csv('predictions.csv')


if __name__ == '__main__':
    main()




