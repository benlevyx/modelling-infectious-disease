import streamlit as st
import csv
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config

def main():
	st.title('Data Preprocessing')

	st.write("The full notebook can be found here: ")

	st.write("### Preprocessing of flu wILI time series data")

	st.write("First we create a dataframe from the raw wILI time series.")

	flu_dir = config.data / 'raw' / 'flu_ground_truth'

	filename = 'wILI_cleaned_Alabama.csv'

	with open(flu_dir / filename, 'r') as f:
	    weeks = [row.split(',')[0] for row in f][1:]
	week2cnt = {weeks[i]: i for i in range(len(weeks))}
	cnt2week = {v: k for k, v in week2cnt.items()}

	week_num = [int(week.split('-')[1]) for week in weeks]

	# create dictionary that maps each state to the corresponding wILI time series
	location2info = {}
	for filename in os.listdir(flu_dir):
	    location = filename.replace('.csv', '').split('_')[-1]
	    with open(flu_dir / filename, 'r') as f:
	        rows = list(csv.reader(f))
	        time2wili = {row[0]: float(row[1]) for row in rows[1:]}
	        location2info[location] = time2wili

	# convert the dictionary to a dataframe
	df = pd.DataFrame.from_dict(location2info)	     

	df.rename(index=week2cnt, inplace=True)
	df.sort_index(inplace=True)  

	# ignore the following locations
	df_51 = df.drop(columns=['Virgin Islands', 'Puerto Rico', 'New York City']) 

	st.write(df_51.head())

	st.write("A small subset of the values in the wILI time series are equal to 0.000100. In the context of the individual time series, it's clear that this value is essentially a missing value. Some states don't have any occurances of this special value while others (usually smaller states such as Montana and North Dakota and Delaware) have dozens of occurances of this value. To address this issue, we decided to impute these missing values for any state whose time series had at least one missing value. Initially, we tried fitting a GAM to the values in each time series that are not missing, but this produced horrible predictions for those weeks with missing values. Next, we tried filling in a given state's missing values with the average of the values of the state's neighbors for the same week, but this also produced bad imputations. Finally, we fit a Gaussian process regression model using the values that weren't missing for each time series and predicted the weeks that had missing values. (Note: because wILI values must be nonnegative, any negative prediction from the Gaussian process was considered as a wILI rate of 0.0.) This resulted in plausible imputations as illustrated in the plot below for North Dakota.")

	df_51_imputed = df_51.copy()
	na_val = 0.0001
	indices = np.array(df_51.index)
	for state in df_51.columns:
	    X = indices.copy()
	    y = np.array(df[state])
	    missing_indices = np.argwhere(y == na_val).flatten()
	    if len(missing_indices):
	        X_train = np.delete(X, missing_indices)
	        y_train = np.delete(y, missing_indices)
	        
	        gpr = GPR().fit(X_train.reshape(-1, 1), y_train)
	        y_preds = gpr.predict(missing_indices.reshape(-1, 1))
	        y_preds = [max(0.0, y_pred) for y_pred in y_preds]
	        
	        for i, missing_index in enumerate(missing_indices):
	            df_51_imputed.iloc[missing_index][state] = y_preds[i]  
	        
	        if state == 'North Dakota':
	            fig = plt.figure(figsize=(20, 10))
	            plt.scatter(X_train, y_train, color='g', label='non-missing values')
	            plt.scatter(missing_indices, y_preds, color='r', label='missing value imputation')
	            plt.xlabel('index')
	            plt.ylabel('wILI')
	            plt.title('wILI time series of ' + state)
	            plt.legend()
	            st.pyplot(fig)


	st.write("The resulting dataframe serves as the raw data for the RNN models.")

	st.write("### Preprocessing for Bayesian Model")

	st.write("For the Bayesian model, we don't want to use the wILI data in its raw form. Instead, we want to calculate the percent change in the wILI rate between consecutive weeks. The reasons for this is explained in the notebook that discusses the Bayesian model.")

	# limit analysis to Lower 48 states
	df_48_imputed = df_51_imputed.drop(columns=['Alaska', 'Hawaii', 'District of Columbia'])

	abbrev2state = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
	}

	state2abbrev = {v: k for k, v in abbrev2state.items()}

	df_48_imputed.columns = [state2abbrev[state] for state in df_48_imputed.columns]

	# calculate the absolute change in the weekly wILI rate for each state. This reduces the number of rows in the 
	# dataframe by 1
	flu_change_df = pd.DataFrame()
	for state in df_48_imputed.columns:
	    wili_vals = df_48_imputed[state]
	    flu_change_df[state] = [abs(wili_vals[i+1] - wili_vals[i]) for i in range(len(wili_vals) - 1)]

	flu_change_df.head()

	st.write("There are two main issues with modelling the percent change in the weekly wILI rates. The first issue is that during the summer months, the wILI rates are generally low, often times close to zero. This means that relatively small changes in the wILI rate can produce massive percent changes (i.e. changing from 0.005 to 0.1 represents an increase of 1900%, while decrease from 0.1 to 0.005 represents a decrease of 95%). These huge spikes, which only reflect minor fluctuations in the wILI rates, can obviously interfere with the inference of the model. Secondly, most of the information that can be used to determine which states are most correlated and which features account for this correlation is confined to the colder months of the year when the wILI rates change significantly on a weekly basis. On the other hand, because the wILI rates are relatively low during the warmer months of the year, these isn't much insight that can be gained during these months. This means that training the model on this data would distract the model from focusing on the informative colder months and could even prevent the model from learning the true correlations if the small fluctuations during the warmer months happen to contradict the true signal during the flu season. Therefore, to avoid these two problems, it makes sense to train the model only on the wILI rates during the flu season. \n\nOne way to decide which weeks of the time series to exclude from the model is to inspect the mean absolute change in the wILI rates of all states during each week. The higher this mean absolute change, the more the wILI rates are varying and the more information about the interstate correlations is present in the data, and vice versa. The plot below displays the mean absolute change in the wILI rates for each week of the time series. The red line signifies a value of 0.25, while the green dashed lines denote where the mean absolute change drops below and rises above this 0.25 threshold. All weeks where the metric is above this 0.25 threshold are included in the model, while those weeks that are below the threshold are excluded. The plot clearly illustrates why a value of 0.25 is a reasonable threshold.")

	changes = flu_change_df.apply(np.mean, axis=1)
	fig = plt.figure(figsize=(20, 10))
	cutoffs = [(0, 26), (56, 90), (108, 132), (162, 177), (212, 235), (265, 290), (321, 344), (370, 393),
	          (422, 446)]
	plt.plot(changes, color='b')
	plt.axhline(y=0.25, color='r', linestyle='--')

	for lower_cutoff, upper_cutoff in cutoffs:
	    plt.axvline(x=lower_cutoff, color='g', linestyle='--')
	    plt.axvline(x=upper_cutoff, color='g', linestyle='--')
	plt.xlabel('week number of wILI time series')
	plt.ylabel('mean absolute change in wILI rate across all states')
	st.pyplot(fig)

	# calculate the percent change in the weekly wILI rate for each state
	flu_percent_change_df = pd.DataFrame()
	for state in df_48_imputed.columns:
	    wili_vals = df_48_imputed[state]
	    flu_percent_change_df[state] = [(wili_vals[i+1] - wili_vals[i]) / wili_vals[i] 
	                                    for i in range(len(wili_vals) - 1)]

	# identify the weeks of the time series to include in the model based on the dashed green lines in the plot above
	indices_to_keep = []
	for lower_cutoff, upper_cutoff in cutoffs:
	    indices_to_keep.extend(list(range(lower_cutoff, upper_cutoff)))   
	original_flu_keep_df = df_48_imputed.iloc[indices_to_keep]	                                    

	st.write("Limiting the analysis to the weeks with a significant average change in wILI rates reduced the number of entries in the percent change dataframe that were either very inflated (i.e. greater than 500%) or very deflated (i.e. lower than -90%) by around 96%. While this was encouraging, we still had to handle the several extreme entries that remained. Comparing the values where the percent change between the two values exceeds either 500 percent or -90 percent to those imputed above reveals that the wILI values that lead to these issues are overwhelmingly those filled in by the Gaussian process model and are therefore inherently unreliable. Therefore, we wouldn't be distorting the original data by changing these values. Because these extreme values comprised just 28 out of the 10199 entries in the percent change dataframe, we used a simple imputation technique that shouldn't have a significant impact on the model -- we simply replaced the extreme entries in the percent change dataframe with the mean percent change of all the other non-extreme entries for the given week.")

	
