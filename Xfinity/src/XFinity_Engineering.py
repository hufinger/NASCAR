import pandas as pd
import numpy as np
from datetime import datetime, date


def fixtracklength(df):
    """
    group tracks with similar lengths or configurations together
    :param df: pd.Dataframe
    :return: corrected df with grouplength column
    """
    for i in range(len(df)):
        if df.length[i] < 1:
            df['grouplength'][i] = 0.5
        elif df.length[i] >= 1 and df.length[i] < 1.2:
            df['grouplength'][i] = 1
        elif df.length[i] == 2.28 or df.track[i] == 'Sonoma' or df.track[i] == 'Watkins Glen' or df.length[i] > 2.7 or \
                df.track[i] == 'Indianapolis G.P.':
            df['grouplength'][i] = 3
        elif df.track[i] == 'Daytona':
            df['grouplength'][i] = 2.7
        elif df.track[i] == 'Darlington' or df.track[i] == 'Nashville':
            df['grouplength'][i] = 1.5
        else:
            df['grouplength'][i] = df.length[i]
    return (df)


def timetosec(df, col):
    """
    change lap times that are in minutes to seconds
    :param df: pd.Dataframe
    :param col: column name
    :return: corrected dataframe with time as sec
    """
    for i in range(df.shape[0]):
        a = str(df.iloc[i, col])
        if a == '':
            a = a.replace('', '0')
        split = a.split(':')
        if split[0] != '1':
            df.iloc[i, col] = float(split[0])
        else:
            time = float(split[0]) * 60 + float(split[1])
            df.iloc[i, col] = time


def career_race(df):
    """
    Get driver career averages at each track
    :param df: pd.Dataframe
    :return: pd.Dataframe
    """
    driver = []
    track = []
    finish = []
    avg_st = []
    years3 = []
    race1 = []
    for i in range(len(df)):
        year = df.year[i]
        race = df.race_num[i]
        grouping_race = race - 1
        track1 = df.track[i]
        if grouping_race != 0:
            row = career[(career['year'] == year) & (career['race_num'] == grouping_race)].index
            data = career[0:row[-1]]
            data = pd.DataFrame(data.groupby(['driver', 'track'], as_index=False)['finish_pos', 'start_pos'].mean())
            data = data[data['track'] == track1]
            for i in range(data.shape[0]):
                driver.append(data.iloc[i, 0])
                track.append(track1)
                finish.append(data.iloc[i, 2])
                avg_st.append(data.iloc[i, 3])
                years3.append(year)
                race1.append(race)
        else:
            row = career[(career['year'] == year - 1)].index
            data = career[0:row[-1]]
            data = pd.DataFrame(data.groupby(['driver', 'track'], as_index=False)['finish_pos', 'start_pos'].mean())
            data = data[data['track'] == track1]
            for i in range(data.shape[0]):
                driver.append(data.iloc[i, 0])
                track.append(track1)
                finish.append(data.iloc[i, 2])
                avg_st.append(data.iloc[i, 3])
                years3.append(year)
                race1.append(race)
    career_avg = pd.DataFrame({
        'driver': driver,
        'track': track,
        'avg_fin': finish,
        'avg_srt': avg_st,
        'year': years3,
        'race_num': race1
    })
    return career_avg


def yearloops(df):
    """
    median loop data by track type
    :param df: pd.Dataframe
    :return: pd.Dataframe
    """
    driver = []
    midrace = []
    high = []
    low = []
    avg = []
    passdif = []
    greenpass = []
    greenpassed = []
    qualpass = []
    fastlap = []
    top15lap = []
    lapled = []
    totallap = []
    driver_rating = []
    year1 = []
    grouplength = []
    racenum = []
    DFS = []
    for i in range(len(df)):
        year = df.year[i]
        race = df.race_num[i]
        grouping_race = race - 1
        glength = looplength[(looplength['year'] == year) & (looplength['race_num'] == race)]
        glength = glength.grouplength.unique()[0]
        if grouping_race > 2 and year >= 2018:
            row = looplength[(looplength['year'] == year) & (looplength['race_num'] == grouping_race)].index
            data = looplength[0:row[-1]]
            data = data[(data.year == year) & (data.grouplength == glength)]
            data = pd.DataFrame(data.groupby(['driver', 'grouplength'], as_index=False).median())
            if data.empty:
                data1 = looplength[0:row[-1]]
                data1 = data1[(data1.year == year - 1) & (data1.grouplength == glength)]
                data1 = pd.DataFrame(data1.groupby(['driver', 'grouplength'], as_index=False).median())
                for i in range(data1.shape[0]):
                    driver.append(data1.iloc[i, 0])
                    midrace.append(data1.iloc[i, 2])
                    high.append(data1.iloc[i, 3])
                    low.append(data1.iloc[i, 4])
                    avg.append(data1.iloc[i, 5])
                    passdif.append(data1.iloc[i, 6])
                    greenpass.append(data1.iloc[i, 7])
                    greenpassed.append(data1.iloc[i, 8])
                    qualpass.append(data1.iloc[i, 9])
                    fastlap.append(data1.iloc[i, 11])
                    top15lap.append(data1.iloc[i, 12])
                    lapled.append(data1.iloc[i, 14])
                    totallap.append(data1.iloc[i, 16])
                    driver_rating.append(data1.iloc[i, 17])
                    DFS.append(data1.iloc[i, 25])
                    year1.append(year)
                    grouplength.append(glength)
                    racenum.append(race)
            for i in range(data.shape[0]):
                driver.append(data.iloc[i, 0])
                midrace.append(data.iloc[i, 2])
                high.append(data.iloc[i, 3])
                low.append(data.iloc[i, 4])
                avg.append(data.iloc[i, 5])
                passdif.append(data.iloc[i, 6])
                greenpass.append(data.iloc[i, 7])
                greenpassed.append(data.iloc[i, 8])
                qualpass.append(data.iloc[i, 9])
                fastlap.append(data.iloc[i, 11])
                top15lap.append(data.iloc[i, 12])
                lapled.append(data.iloc[i, 14])
                totallap.append(data.iloc[i, 16])
                driver_rating.append(data.iloc[i, 17])
                DFS.append(data.iloc[i, 25])
                year1.append(year)
                grouplength.append(glength)
                racenum.append(race)
        elif grouping_race <= 2 and year > 2018:
            row = looplength[(looplength['year'] == year - 1)].index
            data1 = looplength[0:row[-1]]
            data1 = data1[(data1.year == year - 1) & (data1.grouplength == glength)]
            data1 = pd.DataFrame(data1.groupby(['driver', 'grouplength'], as_index=False).median())
            for i in range(data1.shape[0]):
                driver.append(data1.iloc[i, 0])
                midrace.append(data1.iloc[i, 2])
                high.append(data1.iloc[i, 3])
                low.append(data1.iloc[i, 4])
                avg.append(data1.iloc[i, 5])
                passdif.append(data1.iloc[i, 6])
                greenpass.append(data1.iloc[i, 7])
                greenpassed.append(data1.iloc[i, 8])
                qualpass.append(data1.iloc[i, 9])
                fastlap.append(data1.iloc[i, 11])
                top15lap.append(data1.iloc[i, 12])
                lapled.append(data1.iloc[i, 14])
                totallap.append(data1.iloc[i, 16])
                driver_rating.append(data1.iloc[i, 17])
                DFS.append(data1.iloc[i, 25])
                year1.append(year)
                grouplength.append(glength)
                racenum.append(race)
    loop_avg = pd.DataFrame({
        'driver': driver,
        'season_mid': midrace,
        'season_high': high,
        'season_low': low,
        'group_len': grouplength,
        'season_avg': avg,
        'seasonpassdif': passdif,
        'seasongreenpass': greenpass,
        'seasongreenpassed': greenpassed,
        'seasonqualitypass': qualpass,
        'seasonfastlap': fastlap,
        'seasontop15': top15lap,
        'seasonlapled': lapled,
        'seasonDR': driver_rating,
        'DFS_avg': DFS,
        'year': year1,
        'race_num': racenum
    })
    return loop_avg


def yearloops1(df):
    """
    median loop data for the entire year
    :param df: pd.Dataframe
    :return: pd.Dataframe
    """
    driver = []
    midrace = []
    high = []
    low = []
    avg = []
    passdif = []
    greenpass = []
    greenpassed = []
    qualpass = []
    fastlap = []
    top15lap = []
    lapled = []
    totallap = []
    driver_rating = []
    year1 = []
    grouplength = []
    racenum = []
    DFS = []
    for i in range(len(df)):
        year = df.year[i]
        race = df.race_num[i]
        grouping_race = race - 1
        glength = looplength[(looplength['year'] == year) & (looplength['race_num'] == race)]
        glength = glength.grouplength.unique()[0]
        if grouping_race > 3 and year >= 2018:
            row = looplength[(looplength['year'] == year) & (looplength['race_num'] == grouping_race)].index
            data = looplength[0:row[-1]]
            data = data[(data.year == year)]
            data = pd.DataFrame(data.groupby(['driver'], as_index=False).median())
            if data.empty:
                data1 = looplength[0:row[-1]]
                data1 = data1[(data1.year == year - 1)]
                data1 = pd.DataFrame(data1.groupby(['driver'], as_index=False).median())
                for i in range(data1.shape[0]):
                    driver.append(data1.iloc[i, 0])
                    midrace.append(data1.iloc[i, 2])
                    high.append(data1.iloc[i, 3])
                    low.append(data1.iloc[i, 4])
                    avg.append(data1.iloc[i, 5])
                    passdif.append(data1.iloc[i, 6])
                    greenpass.append(data1.iloc[i, 7])
                    greenpassed.append(data1.iloc[i, 8])
                    qualpass.append(data1.iloc[i, 9])
                    fastlap.append(data1.iloc[i, 11])
                    top15lap.append(data1.iloc[i, 12])
                    lapled.append(data1.iloc[i, 14])
                    totallap.append(data1.iloc[i, 16])
                    driver_rating.append(data1.iloc[i, 17])
                    DFS.append(data1.iloc[i, 25])
                    year1.append(year)
                    grouplength.append(glength)
                    racenum.append(race)
            for i in range(data.shape[0]):
                driver.append(data.iloc[i, 0])
                midrace.append(data.iloc[i, 2])
                high.append(data.iloc[i, 3])
                low.append(data.iloc[i, 4])
                avg.append(data.iloc[i, 5])
                passdif.append(data.iloc[i, 6])
                greenpass.append(data.iloc[i, 7])
                greenpassed.append(data.iloc[i, 8])
                qualpass.append(data.iloc[i, 9])
                fastlap.append(data.iloc[i, 11])
                top15lap.append(data.iloc[i, 12])
                lapled.append(data.iloc[i, 14])
                totallap.append(data.iloc[i, 16])
                driver_rating.append(data.iloc[i, 17])
                DFS.append(data.iloc[i, 25])
                year1.append(year)
                grouplength.append(glength)
                racenum.append(race)
        elif grouping_race <= 3 and year > 2018:
            row = looplength[(looplength['year'] == year - 1)].index
            data1 = looplength[0:row[-1]]
            data1 = data1[(data1.year == year - 1)]
            data1 = pd.DataFrame(data1.groupby(['driver'], as_index=False).median())
            for i in range(data1.shape[0]):
                driver.append(data1.iloc[i, 0])
                midrace.append(data1.iloc[i, 2])
                high.append(data1.iloc[i, 3])
                low.append(data1.iloc[i, 4])
                avg.append(data1.iloc[i, 5])
                passdif.append(data1.iloc[i, 6])
                greenpass.append(data1.iloc[i, 7])
                greenpassed.append(data1.iloc[i, 8])
                qualpass.append(data1.iloc[i, 9])
                fastlap.append(data1.iloc[i, 11])
                top15lap.append(data1.iloc[i, 12])
                lapled.append(data1.iloc[i, 14])
                totallap.append(data1.iloc[i, 16])
                driver_rating.append(data1.iloc[i, 17])
                DFS.append(data1.iloc[i, 25])
                year1.append(year)
                grouplength.append(glength)
                racenum.append(race)
    loop_avg = pd.DataFrame({
        'driver': driver,
        'season_long_mid': midrace,
        'season_long_high': high,
        'season_long_low': low,
        'season_long_avg': avg,
        'seasonlongpassdif': passdif,
        'seasonlonggreenpass': greenpass,
        'seasonlonggreenpassed': greenpassed,
        'seasonlongqualitypass': qualpass,
        'seasonlongfastlap': fastlap,
        'seasonlongtop15': top15lap,
        'seasonlonglapled': lapled,
        'seasonlongDR': driver_rating,
        'DFS_avg_season': DFS,
        'year': year1,
        'race_num': racenum
    })
    return loop_avg


def career_starts(df):
    """
    gets number of starts in the series for the driver
    :param df: pd.Dataframe
    :return: pd.Dataframe containing number of starts for each driver
    """
    start_num = []
    race_num = []
    years = []
    drivers = []
    for i in range(len(df)):
        year = df.year[i]
        race = df.race_num[i]
        row = career[(career.year == year) & (career.race_num == race - 1)].index
        if row.empty:
            row = career[(career.year == year - 1)].index
        data = career[0:row[-1]]
        data = data.groupby('driver').count()
        data = data.reset_index(level=0)
        for driver in data.driver:
            data1 = data[data.driver == driver].reset_index()
            start_num.append(data1.loc[0, 'track'])
            race_num.append(race)
            years.append(year)
            drivers.append(driver)

    career_start = pd.DataFrame({'num_starts': start_num,
                                 'race_num': race_num,
                                 'year': years,
                                 'driver': drivers})
    return career_start


def encode_and_bind(original_dataframe, feature_to_encode):
    """
    one-hot encode and drop original feature
    :param original_dataframe: pd.Dataframe
    :param feature_to_encode: string of column name
    :return: pd.Dataframe
    """
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop(res[[feature_to_encode]], axis=1)
    return (res)


if __name__ == "__main__":
    schedules = pd.read_csv('../data/raw/XFinity_schedule.csv', index_col=[0])  # joined
    print(schedules.head())
    career = pd.read_csv('../data/raw/XFinity_driver_career.csv', index_col=[0])  # joined avg
    entry = pd.read_csv('../data/raw/XFinity_entry.csv', index_col=[0])  # joined
    loop = pd.read_csv('../data/raw/XFinity_loop.csv', index_col=[0])  # joined
    qual = pd.read_csv('../data/raw/XFinity_qual.csv', index_col=[0])  # joined
    # practice = pd.read_csv('XFinity_practice.csv', index_col=[0])

    joiner = schedules[['length', 'year', 'race_num', 'track']]
    looplength = pd.merge(loop, joiner, on=['year', 'race_num'], how='outer')
    looplength['grouplength'] = 0.0

    looplength = fixtracklength(looplength)

    career['driver'] = career['driver'].str.lstrip()
    entry['driver'] = entry['driver'].str.lstrip()
    entry['owner'] = entry['owner'].str.lstrip()
    entry['crew_chief'] = entry['crew_chief'].str.lstrip()
    # practice['driver'] = practice['driver'].str.lstrip()
    loop['driver'] = loop['driver'].str.lstrip()
    qual['driver'] = qual['driver'].str.lstrip()
    qual['qual_time'] = qual['qual_time'].str.strip()
    # practice['prac_time'] = practice['prac_time'].str.strip()
    # practice['prac_diff'] = practice['prac_diff'].str.strip()
    career = career.sort_values(by=['year', 'race_num'])
    career = career.reset_index(drop=True)
    looplength = looplength.sort_values(by=['year', 'race_num'])
    looplength = looplength.reset_index(drop=True)

    looplength['grouplength'] = round(looplength['grouplength'], 1)
    thisyear = int(date.today().year)

    timetosec(qual, qual.columns.get_loc('qual_time'))
    # timetosec(practice, practice.columns.get_loc('prac_time'))
    # timetosec(practice, practice.columns.get_loc('prac_diff'))
    qual.qual_time = pd.to_numeric(qual['qual_time'], downcast='float')
    # practice.prac_time = pd.to_numeric(practice['prac_time'], downcast='float')
    # practice.prac_diff = pd.to_numeric(practice['prac_diff'], downcast='float')

    loop_avg = yearloops(schedules)
    loop_avg1 = yearloops1(schedules)

    starts = career_starts(schedules)

    for i in range(schedules.shape[0]):
        schedules.iloc[i, 4] = datetime.strptime(schedules.iloc[i, 4], '%Y-%m-%d')
    # career_avg = career_race(schedules)

    # print(career_avg)

    test = schedules.loc[[20]]
    test1 = entry[(entry['year'] == 2018) & (entry['race_num'] == 21)]

    test2 = pd.merge(test1, test, how='outer')
    test3 = pd.merge(schedules, entry, how='outer')

    test4 = qual[(qual['race_num'] == 21) & (qual['year'] == 2018)]
    test5 = pd.merge(test3, qual, how='outer', on=['race_num', 'year', 'driver'])

    test6 = pd.DataFrame(career[['driver', 'race_num', 'year', 'finish_pos']])

    test7 = pd.merge(test6, test5, on=['race_num', 'year', 'driver'], how='outer')

    # test8 = pd.merge(test7, career_avg, on = ['race_num', 'year', 'driver', 'track'], how = 'outer')

    test9 = pd.merge(test7, loop_avg, on=['year', 'race_num', 'driver'], how='outer')
    test9_1 = pd.merge(test9, loop_avg1, on=['year', 'race_num', 'driver'], how='outer')
    test9_2 = pd.merge(test9_1, starts, on=['year', 'race_num', 'driver'], how='outer')
    DFS_loop = loop[['DFS', 'DfsRank', 'race_num', 'year', 'driver']]
    test10 = pd.merge(test9_2, DFS_loop, on=['year', 'race_num', 'driver'], how='outer')
    # test10 = test10.dropna().reset_index()

    print(test10.groupby('owner')['finish_pos'].mean())

    # Fix Stewart Haas Owner problem
    for i in range(len(test10)):
        if test10.owner[i] == 'Mike Curb':
            test10.owner[i] = 'Duke Thorson'
    covidfriendly = test10.drop(['qual_time', 'driver_link'], axis=1)

    covidfriendly['top3'] = np.where(covidfriendly.finish_pos <= 3, 1, 0)
    covidfriendly['top5'] = np.where(covidfriendly.finish_pos <= 5, 1, 0)
    covidfriendly['top10'] = np.where(covidfriendly.finish_pos <= 10, 1, 0)

    data = covidfriendly

    data = data.sort_values(by=['year', 'race_num', 'qual_pos'])
    data = data.reset_index(drop=True)

    data = data.dropna(subset=['owner', 'track', 'crew_chief', 'car_make'])
    data = data.reset_index(drop=True)

    DFS_data = data.drop(['date', 'track', 'length', 'crew_chief', 'number'], axis=1)

    cutoff = data.groupby('owner')['finish_pos'].mean().reset_index()
    cutoff = cutoff[cutoff.finish_pos < 20]
    print(cutoff)
    data = data[(data.owner.isin(cutoff.owner))]
    data = data[data.owner != 'Ronnie Bassett, Jr.']
    no_owner = ['SS-GLR w/Jeff Lefcourt', 'Ronnie Bassett, Jr.', 'Sam Hunt Racing', 'Victor Obaika',
                'SS Green Light Racing', 'Chris Cockrum', 'Al Niece', 'Emerling-Gase Motorsports',
                'Reaume Brothers Racing', 'Jesse Iwuji', 'Our Motorsports', 'Michael Contarino']
    DFS_data = DFS_data[~DFS_data.isin(no_owner)]
    print(data)
    print(data.groupby('owner')['finish_pos'].mean())

    data = data.reset_index(drop=True)

    data = encode_and_bind(data, 'car_make')
    data = encode_and_bind(data, 'owner')
    DFS_data = encode_and_bind(DFS_data, 'owner')
    DFS_data = encode_and_bind(DFS_data, 'car_make')

    row = data.index[-1]
    row = data.iloc[row, :]
    race = row.race_num
    year = row.year

    next_race = data[(data.race_num == race) & (data.year == year)]
    DFS_next = DFS_data[(DFS_data.race_num == race) & (DFS_data.year == year)]
    data = data[data.year != thisyear]
    DFS_data = DFS_data[DFS_data.year != thisyear]

    data = data.drop(['date', 'track', 'length', 'crew_chief', 'number'], axis=1)
    next_race = next_race.drop(['date', 'track', 'length', 'crew_chief', 'number'], axis=1)

    h2h = pd.merge(data, data, on=['race_num', 'year'], how='outer')
    h2h_next = pd.merge(next_race, next_race, on=['race_num', 'year'], how='outer')

    h2h = h2h[h2h['driver_x'] != h2h['driver_y']]
    h2h_next = h2h_next[h2h_next['driver_x'] != h2h_next['driver_y']]

    h2h['top_finish'] = np.where(h2h.finish_pos_x < h2h.finish_pos_y, 1, 0)
    h2h = h2h.drop(
        ['year', 'driver_x', 'driver_y', 'top3_x', 'top3_y', 'top5_y', 'top5_x', 'top10_y', 'top10_x',
         'finish_pos_x', 'finish_pos_y', 'DFS_x', 'DFS_y', 'DfsRank_x', 'DfsRank_y'], axis=1)
    h2h = h2h.drop(columns=['race_num'])

    data = data.drop(['race_num', 'year', 'driver'], axis=1)
    DFS_data = DFS_data.drop(['race_num', 'year', 'driver'], axis=1)
    DFS_next = DFS_next.drop(['race_num', 'year'], axis=1)
    next_race = next_race.drop(['race_num', 'year'], axis=1)
    h2h_next = h2h_next.drop(['year', 'race_num'], axis=1)

    data = data.dropna()
    data = data.reset_index(drop=True)

    DFS_data = DFS_data.dropna()
    DFS_data = DFS_data.reset_index(drop=True)

    h2h = h2h.dropna()
    h2h = h2h.reset_index(drop=True)

    next_race = next_race.dropna(subset=['qual_pos'])  # removed avg_fin
    next_race = next_race.drop(['finish_pos', 'top3', 'top5', 'top10', 'DFS', 'DfsRank'], axis=1)
    next_race = next_race.dropna()

    DFS_data = DFS_data.dropna(subset=['qual_pos'])  # removed avg_fin
    DFS_data = DFS_data.drop(['finish_pos', 'top3', 'top5', 'top10'], axis=1)
    DFS_data = DFS_data.dropna()

    DFS_next = DFS_next.dropna(subset=['qual_pos'])  # removed avg_fin
    DFS_next = DFS_next.drop(['finish_pos', 'top3', 'top5', 'top10', 'DFS', 'DfsRank'], axis=1)
    DFS_next = DFS_next.dropna()

    h2h_next = h2h_next.dropna(subset=['qual_pos_x', 'qual_pos_y'])  # removed avg_fin_x and avg_fin_y
    h2h_next = h2h_next.drop(
        ['finish_pos_x', 'top3_x', 'top5_x', 'finish_pos_y', 'top3_y', 'top5_y', 'top10_x', 'top10_y', 'DFS_x', 'DFS_y',
         'DfsRank_x', 'DfsRank_y'], axis=1)
    h2h_next = h2h_next.dropna()

    data.to_csv('../data/processed/covidXFinity.csv')
    DFS_data.to_csv('../data/processed/DFS_XFinity.csv')
    DFS_next.to_csv('../data/next_race/DFS_next_XFinity.csv')
    next_race.to_csv('../data/next_race/next_race_XFinity.csv')
    h2h.to_csv('../data/processed/H2H-XFinity.csv')
    h2h_next.to_csv('../data/next_race/H2H_next_XFinity.csv')
