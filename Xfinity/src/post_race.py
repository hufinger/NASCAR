import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

def join_preds(df, predictions):
    """
    Get model performance metrics on previous race
    :param df: dataframe of the prior race's results
    :param predictions: dataframe of model predictions
    :return: printed model metrics
    """
    first = pd.merge(df, predictions, left_on='Driver', right_on='driver1_x', how='outer')
    second = pd.merge(df, first, left_on='Driver', right_on='driver2_x', how='outer')

    second = second.dropna()

    second['actual_top_finish'] = np.where(second.Finish_x > second.Finish_y, 1, 0)
    second['predicted_correct'] = np.where(second.prediction_x == second.actual_top_finish, 1, 0)

    print(second.predicted_correct.mean())
    print(second.groupby('tier')['predicted_correct'].size())
    print(second.groupby('tier')['predicted_correct'].mean())

    h2h_odds = pd.read_csv('~/Desktop/Projects/NASCAR/Xfinity/data/prediction/odds7.csv', index_col=[0])
    print(df.columns)
    h2h_odds = pd.merge(h2h_odds, df, right_on=['Driver'], left_on=['driver1'], how='left')
    h2h_odds = pd.merge(h2h_odds, df, right_on=['Driver'], left_on=['driver2'], how='left')
    test = pd.merge(h2h_odds, predictions, left_on=['driver1', 'driver2'], right_on=['driver1_x', 'driver2_x'],
                    how='right').dropna()

    test['actual_top_finish'] = np.where(test.Finish_x > test.Finish_y, 1, 0)
    test['predicted_correct'] = np.where(test.prediction_x == test.actual_top_finish, 1, 0)

    print(test.predicted_correct.mean())
    # print(second.groupby('tier')['predicted_correct'].size())
    print(test.groupby('tier')['predicted_correct'].mean())


def get_results(race_num, predictions):
    """
    Get results from the previous race to evaluate model performance
    :param race_num: string of the race number in XX format (ex: 7th race is "07")
    :param predictions: dataframe of the model's race predictions
    :return: N/A
    """
    finish = []
    start = []
    driver = []
    led = []
    url = requests.get('https://www.racing-reference.info/race-results/2022-' + race_num + '/B/').text
    soup = BeautifulSoup(url, 'html.parser')
    my_table = soup.find('table', class_='tb race-results-tbl')
    my_table1 = my_table.findAll('tr', class_="odd")
    my_table2 = my_table.findAll('tr', class_='even')
    for i in range(len(my_table2)):
        a = my_table2[i].findAll("td")
        finish.append(int(a[0].text))
        start.append(int(a[1].text))
        driver.append(a[3].text.lstrip().rstrip())
        led.append(int(a[8].text))
    for i in range(len(my_table1)):
        a = my_table1[i].findAll("td")
        finish.append(int(a[0].text))
        start.append(int(a[1].text))
        driver.append(a[3].text.lstrip().rstrip())
        led.append(int(a[8].text))
    results = pd.DataFrame({'Driver': driver,
                            'Start': start,
                            'Finish': finish,
                            'laps_led': led})
    join_preds(results, predictions)


if __name__ == '__main__':
    race_num = input('Race Number?: ')
    preds = pd.read_csv('../data/prediction/h2h_predictions.csv')

    get_results(race_num, preds)




