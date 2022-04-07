import pandas as pd
import pulp
from pulp import PULP_CBC_CMD


def get_lineups(df, k):
    """
    Getting the optimized DFS lineups from the predictions
    :param df: Pandas Dataframe with predictions, salaries, & driver name
    :param k: int - number of lineups required
    :return: printed list of top k lineups for that week's DFS tournaments
    """
    drivers = list(df.Driver)
    salary = dict(zip(drivers, df['Salary']))
    predictions = dict(zip(drivers, df['Expected']))
    merged['position'] = "Driver"
    position = dict(zip(drivers, df['position']))

    player_vars = pulp.LpVariable.dicts("Drivers", drivers, lowBound=0, upBound=1, cat=pulp.LpInteger)
    total_score = pulp.LpProblem("Fantasy_Points_Problem", pulp.LpMinimize)
    total_score += pulp.lpSum([predictions[i] * player_vars[i] for i in player_vars])
    total_score += pulp.lpSum([salary[i] * player_vars[i] for i in player_vars]) <= 50 * 1000
    pg = [p for p in position.keys() if position[p] == 'Driver']
    total_score += pulp.lpSum([player_vars[i] for i in pg]) == 6

    total_score.solve(PULP_CBC_CMD(msg=0))

    for v in total_score.variables():
        if v.varValue > 0:
            print(v.name)
    print('\n')

    for i in range(1, k):

        a = [v.varValue for v in total_score.variables()]
        b = [v for v in total_score.variables()]

        total_score += pulp.lpSum([a[i] * b[i] for i in range(len(a))]) <= 5.9

        total_score.solve(PULP_CBC_CMD(msg=0))

        for v in total_score.variables():
            if v.varValue > 0:
                print(v.name)

        print('\n')


if __name__ == '__main__':
    salary = pd.read_csv('../data/prediction/dfs_salary7.csv', index_col=[0])
    predictions = pd.read_csv('../data/prediction/dfs_predictions.csv', index_col=[0])

    merged = pd.merge(salary, predictions, on='Driver', how='right')
    merged.Salary = merged.Salary.str.replace(',', '')
    merged = merged.dropna().reset_index(drop=True)
    merged.Salary = merged.Salary.astype(int)

    get_lineups(df=merged, k=3)
