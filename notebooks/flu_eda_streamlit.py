import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import csv
plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config


st.title('Flu Data EDA')

st.write('## Authors: Dimitris Vamvourellis, Benjamin Levy, Will Fried, Matthieu Meeus')


st.write('## Data Loading')
st.write('First we will load the data from all states into one single dataframe. Basically, for each state\
	      we have the weekly influenza-like illness (WILI) rate for the last 10 years.\
	     ')

@st.cache
def fetch_and_clean_data():
	flu_dir = config.data / 'raw' / 'flu_ground_truth'

	#flu_dir = '../flu_ground_truth/'
	filename = 'wILI_cleaned_Alabama.csv'
	with open(flu_dir / filename, 'r') as f:
	    weeks = [row.split(',')[0] for row in f][1:]
	week2cnt = {weeks[i]: i for i in range(len(weeks))}

	location2info = {}
	for filename in os.listdir(flu_dir):
	    location = filename.replace('.csv', '').split('_')[-1]
	    with open(flu_dir / filename, 'r') as f:
	        rows = list(csv.reader(f))
	        time2wili = {row[0]: float(row[1]) for row in rows[1:]}
	        location2info[location] = time2wili

	df = pd.DataFrame.from_dict(location2info)

	df.rename(index=week2cnt, inplace=True)
	df.sort_index(inplace=True)

	#fill missing values
	df = df.fillna(df.mean())
	return df 

df = fetch_and_clean_data()

st.write(df)

st.write('## WILI rate through time in each state')
st.write("""
	      Below we can see the WILI rate time series (first plot) for each state as well as the mean
	      WILI rate over the states at each point in time. It is evident that there is a clear seasonal
	      pattern with the peak value varying by state.
	    """)


if st.checkbox('Show chart'):
	fig = plt.figure(figsize=(20, 10))
	for col in df.columns:
		data = df.loc[df[col].notnull(), col]
		plt.plot(list(data.index), list(data))
	st.pyplot(fig)

month_avg = []
for idx in df.index:
    month_data = df.loc[idx, :]
    month_data = [x for x in month_data if not pd.isnull(x)]
    month_avg.append(np.mean(month_data))

fig = plt.figure(figsize=(20, 10))
plt.plot(df.index, month_avg)
plt.xlabel('weeks since 2010-40')
plt.ylabel('')
st.pyplot(fig)

st.write("""
	      Below we can see the WILI rate time series for any combination of states to examine any patterns between specific states.
	    """)

column = st.multiselect('Select states to display', options=df.columns)
st.line_chart(df[column])

correlation_df = df.corr()

st.write('## Correlation Analysis')
st.write("""
	     To better understand whether there are any patterns related to the location of states, below we calculate the correlaiton matrix 
	     between states based on the WILI state time series. 
	    """)

correlation_df

states_lat_long = pd.read_csv(config.data / 'raw' / 'statelatlong.csv')
states_lat_long = states_lat_long.rename(columns={"Latitude": "latitude", "Longitude": "longitude", "City":"State_full"})


st.write('To visualize the correlations between states, below you can choose a specific state and see a map of the correlation of its WILI time series\
	with any other state. By looking at a few different states, it is clear that most states are strongly correlated with most of \
	of the other states in US without necessarily being more correlated with their adjacent ones. However, there are some exceptions like California\
	which seems to be a lot more correlated to its adjacent states rather than states on the East Coast for example.')


column = st.selectbox('Select state that you want to see correlations with respect to.',
     df.columns)
corr_state_df = pd.DataFrame(correlation_df.loc[column]).reset_index().rename(columns={'index':'State_full', column:'Correlation'})
state_corr_merged = pd.merge(states_lat_long, corr_state_df, how='inner', on='State_full')

fig = px.choropleth(state_corr_merged,  # Input Pandas DataFrame
                    locations="State",  # DataFrame column with locations
                    color='Correlation',  # DataFrame column with color values
                    hover_name="State", # DataFrame column hover info
                    locationmode = 'USA-states') # Set to plot as US States
fig.update_layout(
    title_text = 'Correlation ', # Create a Title
    geo_scope='usa',  # Plot only the USA instead of globe
)

st.plotly_chart(fig)


st.write('Finally, below we can see the mean WILI rate for each state across the last 10 years. This indicates\
	that the the northern states seem to have lower mean WILI rates than the southern states.')



mean_wili_df = pd.DataFrame(df.mean(), columns=['Mean WILI']).reset_index()
mean_wili_df = mean_wili_df.rename(columns={'index':'State_full'})

mean_wili_by_state_df = pd.merge(states_lat_long, mean_wili_df, how='inner', on='State_full')

fig = px.choropleth(mean_wili_by_state_df,  # Input Pandas DataFrame
                    locations="State",  # DataFrame column with locations
                    color="Mean WILI",  # DataFrame column with color values
                    hover_name="State", # DataFrame column hover info
                    locationmode = 'USA-states') # Set to plot as US States
fig.update_layout(
    title_text = 'Mean WILI by State', # Create a Title
    geo_scope='usa',  # Plot only the USA instead of globe
)

st.plotly_chart(fig)

