import pandas as pd
import numpy as np
import streamlit as st
import shap
import pickle

data = pd.read_csv('~/Desktop/Projects/NASCAR/Cup/data/prediction/h2h_predictions.csv', index_col=[0])
data = data.reset_index(drop=True)
finish = pd.read_csv('~/Desktop/Projects/NASCAR/Cup/data/prediction/FinishPos.csv', index_col=[0])
dfs = pd.read_csv('~/Desktop/Projects/NASCAR/Cup/data/prediction/dfs_predictions.csv', index_col=[0])


st.title("NASCAR Head-to-Head Prediction App")
st.set_option('deprecation.showPyplotGlobalUse', False)

with open('/Users/hufinger/Desktop/Projects/NASCAR/Cup/prediction_artifacts/predicted_h2h_shap.pkl', 'rb') as file:
    shap_values = pickle.load(file)

st.write("Please select the two drivers from the drop down menus below:")
driver1 = st.selectbox("Driver 1", (np.unique(data.driver1_x)))
driver2 = st.selectbox("Driver 2", (np.unique(data.driver2_x)))

if driver1 == driver2:
    st.write('Please select two unique drivers.')
else:
    data1 = data[(data.driver1_x == driver1) & (data.driver2_x == driver2)]
    data1 = data1.reset_index(drop=True)
    if data1.empty:
        st.write('This one is too close to call. It is recommended to avoid betting this match-up.')
    else:
        result = data1.prediction_x[0]
        chance = data1.avg_driver1[0]*100
        chance2 = data1.avg_driver2[0]*100
        tier = data1.tier[0]

        st.write('According to our model, ' + driver1 + ' has a ' + str(round(chance,2)) +
                 '% likelihood of being the higher finisher in this match-up.')
        st.write('According to our model, ' + driver2 + ' has a ' + str(round(chance2,2)) +
                 '% likelihood of being the higher finisher in this match-up.')
        if chance > chance2:
            st.write('We recommend betting on ' + driver1 + ' to win with a ' + tier + ' markup.')
        else:
            st.write('We recommend betting on ' + driver2 + ' to win with a ' + tier + ' markup.')
        st.write('The justification can be seen in the graph below.')
        shap.plots.waterfall(shap_values[data1.loc[0,'index_x']], max_display=15)
        st.pyplot(bbox_inches='tight')

st.title('Driver in Top 3, 5, or 10')
st.write("Please select the driver from the drop down menus below:")
driver3 = st.selectbox("Driver:", (np.unique(finish.Driver)))
finish1 = finish[finish.Driver == driver3].reset_index()
st.write(finish[['Driver', 'Top3', 'Top5', 'Top10']])

st.title('DFS Predictions')
st.write('A "Predicted Group" of 1 means the driver is predicted in the top 4 DFS scorers.')
st.write('A "Predicted Group" of 2 means the driver is predicted 5th to 12th in DFS scoring.')
st.write('A "Predicted Group" of 3 means the driver is predicted 13th to 20th in DFS scoring.')
st.write('A "Predicted Group" of 4 means the driver is predicted outside the top 20 DFS scorers.')

dfs = dfs.sort_values('Expected').reset_index().drop('index', axis = 1)
dfs.index += 1
st.write(dfs)
st.write('Not all predicted groups will fit the original parameters. For example group 1 may only have 3 drivers.')
st.write('This means the model is not confident in which driver fills the empty spot. '
         'It thinks the probability of the driver finishing in a different group is higher.')

h2h_odds = pd.read_csv('~/Desktop/Projects/NASCAR/Cup/data/prediction/odds7.csv', index_col=[0])
test = pd.merge(h2h_odds, data, left_on=['driver1', 'driver2'], right_on=['driver1_x', 'driver2_x'], how='right').dropna()
st.write(test[['driver1', 'driver2', 'prediction_x', 'tier', 'book', 'odds1', 'odds2', 'avg_driver1', 'avg_driver2']])
