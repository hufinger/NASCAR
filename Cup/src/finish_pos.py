import math

import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from lazypredict.Supervised import LazyClassifier

from sklearn.preprocessing import MinMaxScaler
import sklearn
import optuna
import numpy as np
import pickle
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix



if __name__ == '__main__':
    #MODEL BUILD IN PROGRESS
    data = pd.read_csv('../data/processed/covidNASCAR.csv', index_col = [0])

    x = data.drop(columns=['top3', 'top5', 'finish_pos', 'top10', 'DFS', 'DfsRank'], axis = 1)
    y = data.finish_pos

    scaler_h2h = MinMaxScaler()
    x = scaler_h2h.fit_transform(x)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.32, random_state=83)

    clf = LazyClassifier(
            ignore_warnings=True, random_state=83862277, verbose=False
        )
    models, predictions = clf.fit(x_train, x_test, y_train, y_test)  # pass all sets

    print(models.head(15))
    print('5 Groups')


    def pct_rank_qcut(series, n):
        edges = pd.Series([float(i) / n for i in range(n + 1)])
        f = lambda x: (edges >= x).argmax()
        print(f)
        return series.rank(pct=1).apply(f)

    def objective(trial):
        data = pd.read_csv('../data/processed/covidNASCAR.csv', index_col=[0])

        x = data.drop(columns=['top3', 'top5', 'top10', 'DFS', 'DfsRank'])
        y = data.finish_pos
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=83, stratify=y)
        param_grid = {
            #         "device_type": trial.suggest_categorical("device_type", ['gpu']),
            'objective': 'multiclass',
            'num_class': 40,
            "n_estimators": trial.suggest_int("n_estimators", 0, 20000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 6000),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 10000),
            "max_bin": trial.suggest_int("max_bin", 0, 500),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.95, step=0.05
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.95, step=0.05
            ),
        }
        # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "softmax")
        bst = lgbm.LGBMClassifier(**param_grid)
        bst.fit(train_x, train_y, eval_set=[(test_x, test_y)], verbose=0)
        preds = bst.predict(test_x)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        return accuracy


    sampler = TPESampler(seed=85)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=50)
    print(study.best_trial.params)

    mod = lgbm.LGBMClassifier(**study.best_trial.params)
    mod.fit(x_train, y_train)
    probs = mod.predict_proba(x_test)
    print(math.sqrt(mean_squared_error(y_test, mod.predict(x_test))))

    for i in range(len(probs)):
        expected = probs[:,i]*(i+1)
    print(expected)

    decile_df = pd.DataFrame({"Probability": probs[:, 1], "Actual": y_test})
    decile_df['Decile'] = pct_rank_qcut(decile_df.Probability, 20)
    print(decile_df.groupby('Decile')['Actual'].mean())
    # print(decile_df.groupby('Decile')['Actual'].size())
    a = decile_df.groupby('Decile')['Probability'].min()
    print(a)