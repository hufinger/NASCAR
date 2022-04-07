import pandas as pd
import numpy as np
import requests
from datetime import date, datetime, timedelta
from bs4 import BeautifulSoup
from time import sleep
from random import randint


def schedule():
    """
    get the schedules, entrylist, qualifying, career, and loop data information for the years in question
    :return: all aforementioned dataframes
    """
    thisyear = int(date.today().year) + 2
    years = [str(i) for i in range(2018, thisyear)]
    race_num = []
    track = []
    length = []
    year_list = []
    race_date = []
    for year in years:
        url = requests.get("https://www.racing-reference.info/season-stats/" + year + "/W/").text
        sleep(randint(2, 5))
        soup = BeautifulSoup(url, 'html.parser')
        my_table = soup.find_all('div', class_="table-row")
        today = date.today() + timedelta(days=1)
        today = today.strftime("%m/%d/%y")
        today = today.strip()
        today = datetime.strptime(today, "%m/%d/%y")
        today = datetime.date(today)
        for i in range(len(my_table)):
            dates = datetime.strptime(my_table[i].find('div', class_="date W").text.strip(), "%m/%d/%y").strftime(
                "%m/%d/%y")
            corrected = datetime.strptime(dates, "%m/%d/%y")
            corrected = datetime.date(corrected)
            if corrected < today:
                race_num.append(int(my_table[i].div.a.text))
                track.append(my_table[i].find('div', class_='track W').text.rstrip())
                length.append(float(my_table[i].find('div', class_='len no-mobile').text))
                year_list.append(int(year))
                race_date.append(corrected)
            if corrected == today:
                race_num.append(int(my_table[i].find('div', class_='race-number').text))
                track.append(my_table[i].find('div', class_='track W').text.rstrip())
                year_list.append(int(year))
                race_date.append(corrected)
                link = my_table[i].find('div', class_='track W').a.get('href')
                url = requests.get(link).text
                soup = BeautifulSoup(url, 'html.parser')
                my_table1 = soup.find_all('tr', class_="odd")
                a = my_table1[1].findAll("td")
                length.append(float(a[6].text))

    schedules = pd.DataFrame({'race_num': race_num,
                              'track': track,
                              'length': length,
                              'year': year_list,
                              'date': race_date})
    schedules["race_num"] = schedules['race_num'].map("{:02}".format)
    print('Schedules Done', datetime.today())
    print(schedules.tail())
    entry, career = entrylist(schedules)
    loop = getloops(schedules)
    print('Loop Data Done', datetime.today())
    qualifying = getqual(schedules)
    print('Qualifying Done', datetime.today())
    loop.to_csv('../data/raw/Cup_loop.csv')
    schedules.to_csv('../data/raw/Cup_schedule.csv')
    qualifying.to_csv('../data/raw/Cup_qual.csv')
    return schedules, entry, career, loop, qualifying


def entrylist(df):
    """
    pull the entry list from racingreference.io for the years in question
    :param df: pd.Dataframe of the schedules
    :return: df with all entry lists
    """
    driver = []
    owner = []
    car = []
    number = []
    CC = []
    links = []
    place = []
    yearseason = []
    for i in range(len(df)):
        race_num = df.race_num[i]
        year = df.year[i].astype(str)
        url = requests.get("https://www.racing-reference.info/entrylist/" + year + '-' + race_num + "/W/E").text
        sleep(randint(2, 5))
        soup = BeautifulSoup(url, 'html.parser')
        my_table = soup.find_all('tr', {'class': "odd"})
        my_table2 = soup.find_all('tr', {'class': "even"})
        for i in range(len(my_table)):
            a = my_table[i].findAll("td")
            driver.append(a[2].text)
            car.append(a[4].text)
            owner.append(a[3].text)
            number.append(a[1].text)
            CC.append(a[5].text)
            links.append(a[2].a.get('href'))
            place.append(race_num)
            yearseason.append(int(year))
        for i in range(len(my_table2)):
            a = my_table2[i].findAll("td")
            driver.append(a[2].text)
            car.append(a[4].text)
            owner.append(a[3].text)
            number.append(a[1].text)
            CC.append(a[5].text)
            links.append(a[2].a.get('href'))
            place.append(race_num)
            yearseason.append(int(year))
    entry_list = pd.DataFrame({'driver': driver,
                               'car_make': car,
                               'owner': owner,
                               'number': number,
                               'crew_chief': CC,
                               'driver_link': links,
                               'race_num': place,
                               'year': yearseason})
    print('Entry List Done', datetime.today())
    driver_list = []
    for i in np.unique(entry_list['driver']):
        driver_list.append(i)
    career_stats = driver_career(driver_list, entry_list)
    print('Career Done', datetime.today())
    entry_list.to_csv('../data/raw/Cup_entry.csv')
    career_stats.to_csv('../data/raw/Cup_driver_career.csv')
    return entry_list, career_stats


def driver_career(driver, entry_list):
    """
    Get driver career information for all races
    :param driver: list of unique driver names
    :param entry_list: pd.Dataframe
    :return: pd.Dataframe of driver career statistics at each track
    """
    track2 = []
    year_race = []
    start = []
    finish = []
    laps = []
    status = []
    driver2 = []
    for names in driver:
        link = entry_list.loc[entry_list['driver'] == names]
        link = np.array(link['driver_link'])
        link = link[0].split('/')[4]
        url = requests.get('https://www.racing-reference.info/rquery?id=' + link + '&trk=t0&series=W').text
        sleep(randint(2, 5))
        soup = BeautifulSoup(url, 'html.parser')
        my_table = soup.find_all('tr', {'class': "odd"})
        my_table2 = soup.find_all('tr', {'class': "even"})
        for i in range(len(my_table)):
            a = my_table[i].findAll("td")
            track2.append(a[1].text.rstrip())
            year_race.append(a[0].text)
            start.append(int(a[3].text))
            finish.append(int(a[4].text))
            laps.append(a[8].text)
            status.append(a[10].text)
            driver2.append(names)
        for i in range(len(my_table2)):
            a = my_table2[i].findAll("td")
            track2.append(a[1].text.rstrip())
            year_race.append(a[0].text)
            start.append(int(a[3].text))
            finish.append(int(a[4].text))
            laps.append(a[8].text)
            status.append(a[10].text)
            driver2.append(names)
    year2 = []
    race_num2 = []
    laps_complete = []
    total_laps = []
    for i in range(len(year_race)):
        split = year_race[i].split('-')
        split2 = laps[i].split('/')
        year2.append(int(split[0]))
        race_num2.append(int(split[1]))
        laps_complete.append(int(split2[0]))
        total_laps.append(int(split2[1]))

    df = pd.DataFrame({
        'driver': driver2,
        'track': track2,
        'start_pos': start,
        'finish_pos': finish,
        'laps_complete': laps_complete,
        'total_laps': total_laps,
        'race_num': race_num2,
        'year': year2,
        'race_status': status
    })
    return df


def getqual(data):
    """
    get qualifying information for each race
    :param data: schedules dataframe
    :return: qualifying dataframe
    """
    races = data['race_num']
    year = data['year'].astype(str)
    driver_qual = []
    qual_pos = []
    time = []
    race_n = []
    season = []
    for years in np.unique(year):
        year = years
        for race in np.unique(races):
            race_num = race
            url = requests.get('https://www.racing-reference.info/getqualify/' + year + '-' + race_num + '/W').text
            sleep(randint(2, 5))
            soup = BeautifulSoup(url, 'html.parser')
            my_table = soup.find_all('tr', {'class': "odd"})
            my_table2 = soup.find_all('tr', {'class': "even"})
            for i in range(len(my_table)):
                a = my_table[i].findAll("td")
                qual_pos.append(int(a[0].text))
                driver_qual.append(a[1].text)
                time.append(a[4].text)
                race_n.append(int(race))
                season.append(int(years))
            for i in range(len(my_table2)):
                a = my_table2[i].findAll("td")
                qual_pos.append(int(a[0].text))
                driver_qual.append(a[1].text)
                time.append(a[4].text)
                race_n.append(int(race))
                season.append(int(years))
    df = pd.DataFrame({
        'qual_pos': qual_pos,
        'driver': driver_qual,
        'qual_time': time,
        'race_num': race_n,
        'year': season
    })
    return (df)


def getpractice(data):
    """
    get practice times
    :param data: schedules dataframe
    :return: practice results dataframe
    """
    races = data['race_num']
    practice = []
    driver_prac = []
    time = []
    diff = []
    prac_laps = []
    rank = []
    year = data['year'].astype(str)
    race_n = []
    season = []
    for years in np.unique(year):
        year = years
        for race in np.unique(races):
            race_num = race
            for j in range(1, 6):
                j = str(j)
                url = requests.get(
                    'https://www.racing-reference.info/getpractice/' + year + '-' + race_num + '/W/' + j).text
                sleep(randint(2, 5))
                soup = BeautifulSoup(url, 'html.parser')
                my_table = soup.find_all('tr', {'class': "odd"})
                my_table2 = soup.find_all('tr', {'class': "even"})
                if not my_table:
                    break
                else:
                    for i in range(len(my_table)):
                        a = my_table[i].findAll("td")
                        practice.append(j)
                        driver_prac.append(a[1].text)
                        time.append(a[4].text)
                        diff.append(a[5].text)
                        prac_laps.append(a[7].text)
                        rank.append(a[0].text)
                        race_n.append(int(race))
                        season.append(int(years))
                    for i in range(len(my_table2)):
                        a = my_table2[i].findAll("td")
                        practice.append(j)
                        driver_prac.append(a[1].text)
                        time.append(a[4].text)
                        diff.append(a[5].text)
                        prac_laps.append(a[7].text)
                        rank.append(a[0].text)
                        race_n.append(int(race))
                        season.append(int(years))
    df = pd.DataFrame({
        'practice_num': practice,
        'driver': driver_prac,
        'prac_time': time,
        'prac_diff': diff,
        'practice_laps': prac_laps,
        'practice_rank': rank,
        'race_num': race_n,
        'year': season
    })
    return (df)


def getloops(data):
    """
    get loop data for each race
    :param data: schedules dataframe
    :return: loop data dataframe
    """
    races = data['race_num']
    year = data['year'].astype(str)
    driver_loop = []
    midrace = []
    high = []
    low = []
    avg = []
    pass_dif = []
    GFPasses = []
    GFPassed = []
    quality_pass = []
    pct_quality_pass = []
    fast_lap = []
    top15 = []
    pct_top15 = []
    laps_led = []
    pct_LL = []
    laps = []
    driver_rating = []
    race_n = []
    season = []
    start = []
    finish = []
    for years in np.unique(year):
        year = years
        for race in np.unique(races):
            race_num = race
            url = requests.get('https://www.racing-reference.info/loopdata/' + year + '-' + race_num + '/W').text
            sleep(randint(2, 5))
            soup = BeautifulSoup(url, 'html.parser')
            my_table = soup.find_all('tr', {'class': "odd"})
            my_table2 = soup.find_all('tr', {'class': "even"})
            for i in range(len(my_table)):
                a = my_table[i].findAll("td")
                driver_loop.append(a[0].text)
                start.append(int(a[1].text))
                midrace.append(int(a[2].text))
                finish.append(int(a[3].text))
                high.append(int(a[4].text))
                low.append(int(a[5].text))
                avg.append(int(a[6].text))
                pass_dif.append(int(a[7].text))
                GFPasses.append(int(a[8].text))
                GFPassed.append(int(a[9].text))
                quality_pass.append(int(a[10].text))
                pct_quality_pass.append(float(a[11].text))
                fast_lap.append(int(a[12].text))
                top15.append(int(a[13].text))
                pct_top15.append(float(a[14].text))
                laps_led.append(int(a[15].text))
                pct_LL.append(float(a[16].text))
                laps.append(int(a[17].text))
                driver_rating.append(float(a[18].text))
                race_n.append(int(race))
                season.append(int(years))
            for i in range(len(my_table2)):
                a = my_table2[i].findAll("td")
                driver_loop.append(a[0].text)
                start.append(int(a[1].text))
                midrace.append(int(a[2].text))
                finish.append(int(a[3].text))
                high.append(int(a[4].text))
                low.append(int(a[5].text))
                avg.append(int(a[6].text))
                pass_dif.append(int(a[7].text))
                GFPasses.append(int(a[8].text))
                GFPassed.append(int(a[9].text))
                quality_pass.append(int(a[10].text))
                pct_quality_pass.append(float(a[11].text))
                fast_lap.append(int(a[12].text))
                top15.append(int(a[13].text))
                pct_top15.append(float(a[14].text))
                laps_led.append(int(a[15].text))
                pct_LL.append(float(a[16].text))
                laps.append(int(a[17].text))
                driver_rating.append(float(a[18].text))
                race_n.append(int(race))
                season.append(int(years))
    df = pd.DataFrame({
        'driver': driver_loop,
        'midrace_pos': midrace,
        'high_pos': high,
        'low_pos': low,
        'avg_pos': avg,
        'pass_dif': pass_dif,
        'GF_Passes': GFPasses,
        'GF_Passed': GFPassed,
        'quality_passes': quality_pass,
        'quality_pass_pct': pct_quality_pass,
        'fast_lap': fast_lap,
        'top15_laps': top15,
        'top15_lap_pct': pct_top15,
        'laps_led': laps_led,
        'LL_pct': pct_LL,
        'total_laps': laps,
        'driver_rating': driver_rating,
        'race_num': race_n,
        'year': season,
        'start': start,
        'finish': finish,
        'laps': laps
    })

    finishpoints = []
    df['pos_dif'] = df.start - df.finish
    for i in range(len(df)):
        if df.finish[i] == 1:
            finishpoints.append(45)
        elif df.finish[i] == 2:
            finishpoints.append(42)
        elif df.finish[i] == 3:
            finishpoints.append(41)
        elif df.finish[i] == 4:
            finishpoints.append(40)
        elif df.finish[i] == 5:
            finishpoints.append(39)
        elif df.finish[i] == 6:
            finishpoints.append(38)
        elif df.finish[i] == 7:
            finishpoints.append(37)
        elif df.finish[i] == 8:
            finishpoints.append(36)
        elif df.finish[i] == 9:
            finishpoints.append(35)
        elif df.finish[i] == 10:
            finishpoints.append(34)
        elif df.finish[i] == 11:
            finishpoints.append(32)
        elif df.finish[i] == 12:
            finishpoints.append(31)
        elif df.finish[i] == 13:
            finishpoints.append(30)
        elif df.finish[i] == 14:
            finishpoints.append(29)
        elif df.finish[i] == 15:
            finishpoints.append(28)
        elif df.finish[i] == 16:
            finishpoints.append(27)
        elif df.finish[i] == 17:
            finishpoints.append(26)
        elif df.finish[i] == 18:
            finishpoints.append(25)
        elif df.finish[i] == 19:
            finishpoints.append(24)
        elif df.finish[i] == 20:
            finishpoints.append(23)
        elif df.finish[i] == 21:
            finishpoints.append(21)
        elif df.finish[i] == 22:
            finishpoints.append(20)
        elif df.finish[i] == 23:
            finishpoints.append(19)
        elif df.finish[i] == 24:
            finishpoints.append(18)
        elif df.finish[i] == 25:
            finishpoints.append(17)
        elif df.finish[i] == 26:
            finishpoints.append(16)
        elif df.finish[i] == 27:
            finishpoints.append(15)
        elif df.finish[i] == 28:
            finishpoints.append(14)
        elif df.finish[i] == 29:
            finishpoints.append(13)
        elif df.finish[i] == 30:
            finishpoints.append(12)
        elif df.finish[i] == 31:
            finishpoints.append(10)
        elif df.finish[i] == 32:
            finishpoints.append(9)
        elif df.finish[i] == 33:
            finishpoints.append(8)
        elif df.finish[i] == 34:
            finishpoints.append(7)
        elif df.finish[i] == 35:
            finishpoints.append(6)
        elif df.finish[i] == 36:
            finishpoints.append(5)
        elif df.finish[i] == 37:
            finishpoints.append(4)
        elif df.finish[i] == 38:
            finishpoints.append(3)
        elif df.finish[i] == 39:
            finishpoints.append(2)
        elif df.finish[i] == 40:
            finishpoints.append(1)

    df['finishpoints'] = finishpoints
    df['DFS'] = df.finishpoints + (.45 * df.fast_lap) + df.pos_dif + (.25 * df.laps_led)
    df.DFS = df.DFS.round(2)

    df['DfsRank'] = df.groupby(['year', 'race_num'])['DFS'].rank('dense', ascending=False).astype(int)
    return df


if __name__ == '__main__':
    print('Start time:', datetime.today())
    schedules, entry, career, loop, qual = schedule()
    # practice = getpractice(schedules)
    # print('Practice Done', datetime.today())
