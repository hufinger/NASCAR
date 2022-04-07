import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

driver = webdriver.Chrome('/Users/hufinger/Downloads/chromedriver100')


def get_dk():
    """
    Get all H2H bets from DraftKings Sportsbook.
    :return: H2H bets dataframe
    """
    pages = ['driver-props', 'featured-matchups']
    driver1 = []
    driver2 = []
    odds1 = []
    odds2 = []
    for page in pages:
        driver.get(
            'https://sportsbook.draftkings.com/leagues/motorsports/88671394?catergory=' + page + '&category=' + page)
        time.sleep(15)
        odds = driver.find_elements_by_class_name("sportsbook-odds.american.default-color")
        drivers = driver.find_elements_by_class_name("sportsbook-outcome-cell__label")
        # print(test[0].find_element_by_class_name('sportsbook-odds.american.default-color').text)
        for i in range(len(odds)):
            if i % 2 == 0:
                driver2.append(drivers[i].text)
                odds2.append(odds[i].text)
            else:
                driver1.append(drivers[i].text)
                odds1.append(odds[i].text)
    dk_final = pd.DataFrame({'driver1': driver1,
                             'odds1': odds1,
                             'driver2': driver2,
                             'odds2': odds2,
                             'book': 'DK'})
    print(dk_final)
    return dk_final


def names_fix(val):
    """
    Fix names with "Jr", so it plays nice with my other data source
    :param val: string with name
    :return: Correct driver name as a string
    """
    val = val.split(',')
    val = val[1] + ' ' + val[0]
    if 'Jr' in val:
        val = val.split(' ')
        val = val[1] + ' ' + val[2] + ',' + ' ' + val[3] + '.'
    val = val.lstrip()
    return val


def get_stool():
    """
    Get all H2H bets from Barstool Sportsbook (I still have  to update event number manually each week.)
    :return: H2H bets dataframe
    """
    driver1 = []
    driver2 = []
    odds1 = []
    odds2 = []
    for page in ['head_to_head', 'featured_matchups']:
        driver.get("https://www.barstoolsportsbook.com/events/1018532170?tab=" + page)
        time.sleep(18)
        drivers = driver.find_elements_by_class_name('desc')
        odds = driver.find_elements_by_class_name('odds')
        for i in range(len(odds)):
            print(i)
            print(drivers[i].text)
            print(odds[i].text)
            if drivers[i].text:
                if i % 2 == 0:
                    driver1.append(drivers[i].text)
                    odds1.append(odds[i].text)
                else:
                    driver2.append(drivers[i].text)
                    odds2.append(odds[i].text)
    stool_final = pd.DataFrame({'driver1': driver1,
                                'odds1': odds1,
                                'driver2': driver2,
                                'odds2': odds2,
                                'book': 'Barstool'})
    stool_final['driver1'] = stool_final['driver1'].apply(names_fix)
    stool_final['driver2'] = stool_final['driver2'].apply(names_fix)
    return stool_final


def get_dfs():
    """
    Scrape the DFS salaries for the weeks Cup race
    :return: Pandas Dataframe containing driver name and salary
    """
    driver1 = []
    salary = []
    driver.get("https://frcs.pro/dfs/draftkings/xfinity/salary-cap-calculator")
    time.sleep(10)
    drivers = driver.find_elements_by_class_name("driver")
    salaries = driver.find_elements_by_class_name("sorting_1")

    for i in range(len(drivers)):
        print(drivers[i].text)
        if "Jr" in drivers[i].text:
            val = drivers[i].text
            val = val.split(' ')
            val = val[0] + ' ' + val[1] + ',' + ' ' + val[2] + '.'
            driver1.append(val)

        else:
            driver1.append(drivers[i].text)

    for i in range(len(salaries)):
        salary.append(salaries[i].text)
    dfs = pd.DataFrame({'Driver': driver1,
                        'Salary': salary})
    return dfs


if __name__ == '__main__':
    draftkings = get_dk()
    barstool = get_stool()
    dfs_salary = get_dfs()
    final = pd.concat([draftkings, barstool], ignore_index=True)
    final.to_csv('../data/prediction/odds8.csv')
    dfs_salary.to_csv('../data/prediction/dfs_salary8.csv')
    print(final)