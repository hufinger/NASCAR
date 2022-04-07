import math

import optuna
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, confusion_matrix, f1_score, \
    mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn
from xgboost import XGBClassifier
import lightgbm as lgbm


def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    print(f)
    return series.rank(pct=1).apply(f)


def top3_refresh():
    """
    Refresh the Top 3 model
    :return: nothing, but creates pickled model object
    """
    data = pd.read_csv('../data/processed/covidXFinity.csv', index_col=[0])

    x = data.drop(columns=['top3', 'top5', 'finish_pos', 'top10', 'DFS', 'DfsRank'])
    print(x.columns)
    y3 = data.top3

    x_train, x_test, y_train, y_test = train_test_split(x, y3, test_size=.33, random_state=83, stratify=y3)

    # clf = LazyClassifier(
    #     ignore_warnings=True, random_state=83862277, verbose=False
    # )
    # models, predictions = clf.fit(x_train, x_test, y_train, y_test)  # pass all sets
    #
    # print(models.head(15))
    # print(y_train.value_counts)

    def objective(trial):
        data = pd.read_csv('../data/processed/covidXFinity.csv', index_col=[0])

        x = data.drop(columns=['top3', 'top5', 'finish_pos', 'top10', 'DFS', 'DfsRank'], axis=1)
        y = data.top3
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=83, stratify=y)

        param = {
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight",
                                                          [1, (train_y.value_counts()[0] / train_y.value_counts()[1])]),
            "n_estimators": trial.suggest_int("n_estimators", 1, 20000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 6000),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 10000),
            "max_bin": trial.suggest_int("max_bin", 2, 500),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
            "subsample": trial.suggest_float(
                "subsample", 0.2, 0.95, step=0.05
            ),
            "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 0.95, step=0.05
            ),
        }
        # Add a callback for pruning.
        mod = lgbm.LGBMClassifier(**param)
        mod.fit(train_x, train_y)
        preds = mod.predict(test_x)
        accuracy = np.mean(cross_val_score(mod, train_x, train_y, cv=5, scoring='f1'))
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=5000)
    print(study.best_trial.params)

    mod = lgbm.LGBMClassifier(**study.best_trial.params)
    min_features_to_select = 30  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=mod,
        step=1,
        cv=StratifiedKFold(),
        scoring="f1",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(x_train, y_train)
    preds = rfecv.predict(x_test)
    probs = rfecv.predict_proba(x_test)
    print(accuracy_score(y_test, rfecv.predict(x_test)))

    print(confusion_matrix(y_test, preds))
    print(confusion_matrix(y_train, rfecv.predict(x_train)))
    print(roc_auc_score(y_test, preds))

    decile_df = pd.DataFrame({"Probability": probs[:, 1], "Actual": y_test})
    decile_df['Decile'] = pct_rank_qcut(decile_df.Probability, 20)
    print('Actuals')
    print(decile_df.groupby('Decile')['Actual'].mean())
    # print(decile_df.groupby('Decile')['Actual'].size())
    a = decile_df.groupby('Decile')['Probability'].min()
    print(a)
    print('top3 done')
    with open('../prediction_artifacts/Top3ModX_1.1.0', 'wb') as file:
        pickle.dump(rfecv, file)


def top5_refresh():
    """
    Refresh the Top 5 model
    :return: nothing, but creates pickled model object
    """
    data = pd.read_csv('../data/processed/covidXFinity.csv', index_col=[0])

    x = data.drop(columns=['top3', 'top5', 'finish_pos', 'top10', 'DFS', 'DfsRank'])
    print(x.columns)
    y5 = data.top5

    x_train, x_test, y_train, y_test = train_test_split(x, y5, test_size=.33, random_state=83, stratify=y5)

    # clf = LazyClassifier(
    #     ignore_warnings=True, random_state=83862277, verbose=False
    #   )
    # models, predictions = clf.fit(x_train, x_test, y_train, y_test)  # pass all sets
    #
    # print(models.head(15))

    def objective(trial):
        data = pd.read_csv('../data/processed/covidXFinity.csv', index_col=[0])

        x = data.drop(columns=['top3', 'top5', 'finish_pos', 'top10', 'DFS', 'DfsRank'], axis=1)
        y = data.top5
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=83, stratify=y)

        param = {
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight",
                                                          [1, (train_y.value_counts()[0] / train_y.value_counts()[1])]),
            "n_estimators": trial.suggest_int("n_estimators", 1, 20000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 6000),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 10000),
            "max_bin": trial.suggest_int("max_bin", 2, 500),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
            "subsample": trial.suggest_float(
                "subsample", 0.2, 0.95, step=0.05
            ),
            "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 0.95, step=0.05
            ),
        }
        # Add a callback for pruning.
        mod = lgbm.LGBMClassifier(**param)
        mod.fit(train_x, train_y)
        preds = mod.predict(test_x)
        accuracy = np.mean(cross_val_score(mod, train_x, train_y, cv=5, scoring='f1'))
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=5000)
    print(study.best_trial.params)

    mod = lgbm.LGBMClassifier(**study.best_trial.params)
    min_features_to_select = 30  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=mod,
        step=1,
        cv=StratifiedKFold(),
        scoring="f1",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(x_train, y_train)
    preds = rfecv.predict(x_test)
    probs = rfecv.predict_proba(x_test)
    print(accuracy_score(y_test, rfecv.predict(x_test)))

    print(confusion_matrix(y_test, preds))
    print(confusion_matrix(y_train, rfecv.predict(x_train)))

    decile_df = pd.DataFrame({"Probability": probs[:, 1], "Actual": y_test})
    decile_df['Decile'] = pct_rank_qcut(decile_df.Probability, 20)
    print(decile_df.groupby('Decile')['Actual'].mean())
    a = decile_df.groupby('Decile')['Probability'].min()
    with open('../prediction_artifacts/Top5ModX_1.1.0', 'wb') as file:
        pickle.dump(rfecv, file)
    print(a)


def top10_refresh():
    """
    Refresh the top 10 model
    :return: nothing, but creates pickled model object
    """
    data = pd.read_csv('../data/processed/covidXFinity.csv', index_col=[0])

    x = data.drop(columns=['top3', 'top5', 'finish_pos', 'top10', 'DFS', 'DfsRank'])
    print(x.columns)
    y10 = data.top10
    x_train, x_test, y_train, y_test = train_test_split(x, y10, test_size=.33, random_state=83, stratify=y10)

    def objective(trial):
        data = pd.read_csv('../data/processed/covidXFinity.csv', index_col=[0])

        x = data.drop(columns=['top3', 'top5', 'finish_pos', 'top10', 'DFS', 'DfsRank'], axis=1)
        y = data.top10
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=83, stratify=y)

        param = {
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight",
                                                          [1, (train_y.value_counts()[0] / train_y.value_counts()[1])]),
            "n_estimators": trial.suggest_int("n_estimators", 1, 20000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 6000),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 10000),
            "max_bin": trial.suggest_int("max_bin", 2, 500),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
            "subsample": trial.suggest_float(
                "subsample", 0.2, 0.95, step=0.05
            ),
            "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 0.95, step=0.05
            ),
        }
        # Add a callback for pruning.
        mod = lgbm.LGBMClassifier(**param)
        mod.fit(x_train, y_train)
        preds = mod.predict(test_x)
        accuracy = np.mean(cross_val_score(mod, train_x, train_y, cv=5, scoring='f1'))
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=5000)
    print(study.best_trial.params)

    mod = lgbm.LGBMClassifier(**study.best_trial.params)
    min_features_to_select = 30  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=mod,
        step=1,
        cv=StratifiedKFold(),
        scoring="f1",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(x_train, y_train)
    preds = rfecv.predict(x_test)
    probs = rfecv.predict_proba(x_test)
    print(accuracy_score(y_test, rfecv.predict(x_test)))

    decile_df = pd.DataFrame({"Probability": probs[:, 1], "Actual": y_test})
    decile_df['Decile'] = pct_rank_qcut(decile_df.Probability, 20)
    print(decile_df.groupby('Decile')['Actual'].mean())
    # print(decile_df.groupby('Decile')['Actual'].size())
    a = decile_df.groupby('Decile')['Probability'].min()
    with open('../prediction_artifacts/Top10ModX_1.1.0', 'wb') as file:
        pickle.dump(rfecv, file)
    print(a)


def h2h_refresh():
    """
    Refresh the Head-to-Head model
    :return: nothing, but creates pickled model object
    """
    data = pd.read_csv('../data/processed/H2H-XFinity.csv', index_col=[0])

    x = data.drop(columns=['top_finish'])
    y = data.top_finish

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=83, stratify=y)

    print('y_train class distribution')
    print(y_train.value_counts(normalize=True))
    print('y_test class distribution')
    print(y_test.value_counts(normalize=True))

    def objective(trial):
        data = pd.read_csv('../data/processed/H2H-XFinity.csv', index_col=[0])

        x = data.drop(columns=['top_finish'])
        y = data.top_finish
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=83, stratify=y)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)

        param = {
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight",
                                                          [1, (train_y.value_counts()[0] / train_y.value_counts()[1])]),
            "n_estimators": trial.suggest_int("n_estimators", 1, 20000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 6000),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 10000),
            "max_bin": trial.suggest_int("max_bin", 2, 500),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
            "subsample": trial.suggest_float(
                "subsample", 0.2, 0.95, step=0.05
            ),
            "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 0.95, step=0.05
            ),
        }
        # Add a callback for pruning.
        mod = lgbm.LGBMClassifier(**param)
        mod.fit(x_train, y_train)
        preds = mod.predict(test_x)
        accuracy = np.mean(cross_val_score(mod, train_x, train_y, cv=5, scoring="accuracy"))
        return accuracy

    sampler = TPESampler(seed=86)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=225)
    print(study.best_trial.params)

    mod = lgbm.LGBMClassifier(**study.best_trial.params)
    min_features_to_select = 30  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=mod,
        step=1,
        cv=StratifiedKFold(),
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(x_train, y_train)
    preds = rfecv.predict(x_test)
    probs = rfecv.predict_proba(x_test)
    print(accuracy_score(y_test, rfecv.predict(x_test)))

    decile_df = pd.DataFrame({"Probability": probs[:, 1], "Actual": y_test})
    decile_df['Decile'] = pct_rank_qcut(decile_df.Probability, 20)
    print(decile_df.groupby('Decile')['Actual'].mean())
    a = decile_df.groupby('Decile')['Probability'].min()
    print(a)

    with open('../prediction_artifacts/H2HX_1.1.0.pkl', 'wb') as file:
        pickle.dump(rfecv, file)


def dfs_refresh():
    """
    Refresh the DFS model
    :return: nothing, but creates pickled model object
    """
    data = pd.read_csv('../data/processed/DFS_XFinity.csv', index_col=[0])
    conditions = [(data.DfsRank <= 4),
                  (data.DfsRank > 4) & (data.DfsRank <= 12),
                  (data.DfsRank > 12) & (data.DfsRank <= 20),
                  (data.DfsRank > 20)]
    values = range(1, 5)
    data['DfsGroup'] = np.select(conditions, values)

    x = data.drop(['DFS', 'DfsRank', 'DfsGroup'], axis=1)
    y_DfsGroup = data.DfsGroup

    x_train, x_test, y_train, y_test = train_test_split(x, y_DfsGroup, test_size=.32, random_state=83)

    print('5 Groups')

    def objective(trial):
        data = pd.read_csv('../data/processed/DFS_XFinity.csv', index_col=[0])
        conditions = [(data.DfsRank <= 4),
                      (data.DfsRank > 4) & (data.DfsRank <= 12),
                      (data.DfsRank > 12) & (data.DfsRank <= 20),
                      (data.DfsRank > 20)]
        values = range(1, 5)
        data['DfsGroup'] = np.select(conditions, values)

        x = data.drop(['DFS', 'DfsRank', 'DfsGroup'], axis=1)
        y_DfsGroup = data.DfsGroup

        train_x, test_x, train_y, test_y = train_test_split(x, y_DfsGroup, test_size=.32, random_state=83)

        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 20000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 6000),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 10000),
            "max_bin": trial.suggest_int("max_bin", 2, 500),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
            "subsample": trial.suggest_float(
                "subsample", 0.2, 0.95, step=0.05
            ),
            "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 0.95, step=0.05
            ),
        }
        # Add a callback for pruning.
        mod = lgbm.LGBMClassifier(**param)
        mod.fit(train_x, train_y)
        preds = mod.predict(test_x)
        accuracy = np.mean(cross_val_score(mod, train_x, train_y, cv=5, scoring='neg_log_loss'))
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=7000)
    print(study.best_trial.params)

    mod = lgbm.LGBMClassifier(**study.best_trial.params)
    min_features_to_select = 30  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=mod,
        step=1,
        cv=StratifiedKFold(),
        scoring="neg_log_loss",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(x_train, y_train)
    preds = rfecv.predict(x_test)

    dfs_probs = rfecv.predict_proba(x_test)
    dfs_exp_value = dfs_probs[:, 0] * 1 + dfs_probs[:, 1] * 2 + dfs_probs[:, 2] * 3 + dfs_probs[:, 3] * 4
    print(accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(math.sqrt(mean_squared_error(y_test, preds)))
    print(math.sqrt(mean_squared_error(y_test, dfs_exp_value)))

    with open('../prediction_artifacts/DFSX_1.1.0.pkl', 'wb') as file:
        pickle.dump(rfecv, file)


if __name__ == '__main__':
    top3_refresh()
    top5_refresh()
    top10_refresh()
    h2h_refresh()
    dfs_refresh()
