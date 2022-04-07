import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt


def tier(df, val1, val2):
    """
    betting tier for h2h results
    :param df: pd.Dataframe
    :param val1: string of column name
    :param val2: string of column name
    :return: string of tier
    """
    v_high = .99
    high = .96
    med = .8
    low = .5
    if df[val1] >= v_high or df[val2] >= v_high:
        return '2x'
    elif df[val1] >= high or df[val2] >= high:
        return '1.5x'
    elif df[val1] >= med or df[val2] >= med:
        return '1x'
    elif df[val1] >= low or df[val2] >= low:
        return '0.5x'


def predict_h2h(h2h):
    """
    predict head-to-head results
    :param h2h: pd.Dataframe
    :return:
    """
    driver1 = h2h.driver_x
    driver2 = h2h.driver_y
    h2h = h2h.drop(['driver_x', 'driver_y'], axis=1)
    h2h = h2h.dropna()
    h2h = h2h.reset_index(drop=True)
    with open('../prediction_artifacts/H2H_2.4.0.pkl', 'rb') as file:
        mod = pickle.load(file)
    percents = mod.predict_proba(h2h)
    y_pred = mod.predict(h2h)
    h2h_result = pd.DataFrame({'driver1': driver1,
                               'driver2': driver2,
                               'prediction': y_pred,
                               'chance': percents[:, 1],
                               'index': h2h.index})
    h2h_result = pd.merge(h2h_result, h2h_result, right_on='driver2', left_on='driver1')

    h2h_result = h2h_result[h2h_result.driver1_y == h2h_result.driver2_x]
    h2h_result = h2h_result.drop(['driver1_y', 'driver2_y'], axis=1)
    h2h_result = h2h_result[h2h_result.prediction_x != h2h_result.prediction_y]
    h2h_result['avg_driver1'] = (h2h_result['chance_x'] + (1 - h2h_result['chance_y'])) / 2
    h2h_result['avg_driver2'] = (h2h_result['chance_y'] + (1 - h2h_result['chance_x'])) / 2
    h2h_result['tier'] = h2h_result.apply(tier, args=('avg_driver1', 'avg_driver2'), axis=1)
    h2h_result.to_csv('../data/prediction/h2h_predictions.csv')


def predict_placements(next):
    """
    predict top 3, 5 and 10 placements
    :param next: pd.Dataframe
    :return:
    """
    driver = next.driver
    next = next.drop(['driver'], axis=1)
    next = next.dropna()
    next = next.reset_index(drop=True)

    with open('../prediction_artifacts/Top3Mod_1.6.0', 'rb') as file:
        mod3 = pickle.load(file)
    with open('../prediction_artifacts/Top5Mod_1.6.0', 'rb') as file:
        mod5 = pickle.load(file)
    with open('../prediction_artifacts/Top10Mod_1.6.0', 'rb') as file:
        mod10 = pickle.load(file)

    percents3 = mod3.predict_proba(next)
    percents5 = mod5.predict_proba(next)
    percents10 = mod10.predict_proba(next)

    finish_pos = pd.DataFrame({'Driver': driver,
                               'Top3Chance': percents3[:, 1],
                               'Top5Chance': percents5[:, 1],
                               'Top10Chance': percents10[:, 1]})

    top3 = 0.65
    top5 = 0.57
    top10 = 0.61

    finish_pos['Top3'] = np.where(finish_pos.Top3Chance >= top3, 1, 0)
    finish_pos['Top5'] = np.where(finish_pos.Top5Chance >= top5, 1, 0)
    finish_pos['Top10'] = np.where(finish_pos.Top10Chance >= top10, 1, 0)

    print(finish_pos)
    finish_pos.to_csv('../data/prediction/FinishPos.csv')


def predict_dfs(dfs_next):
    """
    predict dfs groups
    :param dfs_next: pd.Dataframe
    :return:
    """
    dfs_driver = dfs_next.driver
    dfs_next = dfs_next.drop(columns=['driver'])

    with open('../prediction_artifacts/DFS_1.5.0.pkl', 'rb') as file:
        dfs = pickle.load(file)

    dfs_preds = dfs.predict(dfs_next)
    dfs_probs = dfs.predict_proba(dfs_next)

    dfs_exp_value = dfs_probs[:, 0] * 1 + dfs_probs[:, 1] * 2 + dfs_probs[:, 2] * 3 + dfs_probs[:, 3] * 4

    dfs_results = pd.DataFrame({'Predicted Group': dfs_preds,
                                'Driver': dfs_driver,
                                'Expected': dfs_exp_value})

    dfs_results.to_csv('../data/prediction/dfs_predictions.csv')


if __name__ == "__main__":
    h2h = pd.read_csv('../data/next_race/H2H_next.csv', index_col=[0])
    next = pd.read_csv('../data/next_race/next_race.csv', index_col=[0])
    dfs_next = pd.read_csv('../data/next_race/DFS_next.csv', index_col=[0])

    predict_h2h(h2h)
    predict_placements(next)
    predict_dfs(dfs_next)
