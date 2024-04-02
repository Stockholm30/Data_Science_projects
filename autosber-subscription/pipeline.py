# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


def main():
    # Use a breakpoint in the code line below to debug your script.
    print('Sbersubscription Prediction Pipeline')  # Press ⌘F8 to toggle the breakpoint.

    df = pd.read_csv('sberauto_train_data')
    x = df.drop(['purpose_action'], axis=1)
    y = df['purpose_action']


    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    models = (
        LogisticRegression(solver = 'liblinear', max_iter = 200, multi_class='ovr'),
        RandomForestClassifier(n_estimators=500, min_samples_split=10),
        SVC()
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, x, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        print(score.mean())
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best_model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    best_pipe.fit(x, y)
    with open('sber_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                "name": "Sbersubscription",
                "author": "Sofya Trifonova",
                "version": 1,

                "type": type(best_pipe.named_steps["classifier"]).__name__,
                "accuracy": best_score
            }
        }, file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
