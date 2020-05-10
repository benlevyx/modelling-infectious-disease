import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import pymc3 as pm 
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config

def main():

	st.title('Flu Inference')

	st.write("The full notebook can be found here: ")

	st.write('## Gathering state-level features')

	st.write("The purpose of this section is to gather state-level features that may affect the degree to which a given state is suspectible or resistant to a virus such as the flu or Covid-19. Collecting these state-level characteristics can help us identify which features are responsible for the correlation in viral infection rates between states, and thus can also be used to quantify the correlation between states based on fundamental attributes of the states rather than just the raw wILI time series.")

	st.write("The density of a state is a natural feature to include because the denser a location, the more easily a virus can spread (look no further than NYC right now). However, it wouldn't make sense to report the density of a state because, for example, the high population density in Manhattan shouldn't be influenced by the fact that upstate New York State has a massive amount of scarsely populated land. Instead, a more sensible measure is a weighted average of the densities of each county in a given state, where the weights are the fraction of the state population that lives in the given county.")

	pred_dir = config.data / 'state_predictors'  

	# dataset that reports the land area in square miles of each county in the U.S.
	land_df = pd.read_csv(pred_dir / 'land_area.csv')

	# dataset that reports the population of each county in the U.S.
	popn_df = pd.read_csv(pred_dir / 'population.csv')


	# st.write(land_df.head())
	# st.write(popn_df.head())

	land_df = land_df[['Areaname', 'LND010190D']]
	popn_df = popn_df[['Areaname', 'PST045200D']]

	# limit analysis to Lower 48 states
	lower_48 = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
            "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

	state_end = tuple(', ' + abbrev for abbrev in lower_48)

	# ignore AL and HI
	filtered_land_df = land_df[land_df.Areaname.str.endswith(state_end)]
	filtered_popn_df = popn_df[land_df.Areaname.str.endswith(state_end)]

	# There are 5 counties in Virginia that are included twice in both the land area and population datasets
	# so we need to ignore the duplicated row
	virginia_counties_df = filtered_land_df[filtered_land_df.Areaname.str.endswith(', VA')]
	indices_to_delete = []
	counties_set = set()
	for index, row in virginia_counties_df.iterrows():
	    county = row['Areaname']
	    if county not in counties_set:
	        counties_set.add(county)
	    else:
	        indices_to_delete.append(index)
	        
	filtered_land_df = filtered_land_df[~filtered_land_df.index.isin(indices_to_delete)]
	filtered_popn_df = filtered_popn_df[~filtered_popn_df.index.isin(indices_to_delete)]

	# merge land area and population datasets
	combined_df = pd.merge(filtered_land_df, filtered_popn_df, on='Areaname', how='inner')

	# extract state from Areaname column
	combined_df['state'] = combined_df.Areaname.str[-2:]
	combined_df.head()

	# rename column names
	combined_df.rename(columns={'Areaname': 'county', 'LND010190D': 'area', 'PST045200D': 'popn'}, inplace=True)

	# fill in missing value of land area of Broomfield, CO from Wikipedia page
	combined_df.loc[combined_df.county == 'Broomfield, CO', 'area'] = 33.00

	# calculate density of each county by dividing population by land area
	combined_df['density'] = combined_df['popn'] / combined_df['area']

	st.write(combined_df.head(10))

	# calculate total population of each state accross all counties
	state2pop = combined_df.groupby('state').agg({'popn': sum}).to_dict()['popn']
	combined_df['state_popn'] = [state2pop[state] for state in combined_df.state]
	combined_df.head()

	# calculate density metric for each state by weighing the density of each population by the fraction of 
	# the state population that lives in the given state
	state2density_metric = (combined_df.groupby('state').
	                        apply(lambda x: round(x['popn'] * (x['density'] ** 1) / x['state_popn'], 1))
	                        .groupby('state').sum()).to_dict()


	# sort states in order of decreasing density
	sorted_density_metrics = sorted(list(state2density_metric.values()), reverse=True)
	density_metric2state = {v: k for k, v in state2density_metric.items()}
	ordered_density_metric2state = {x: density_metric2state[x] for x in sorted_density_metrics}

	# create dataframe with this first state-level feature
	state_stats_df = pd.DataFrame(ordered_density_metric2state.keys(), columns=['density_metric'], 
	                              index=ordered_density_metric2state.values())


	st.write(state_stats_df)

	st.write("The next feature is the average latitude of each state.")

	latlong_df = pd.read_csv(pred_dir / 'statelatlong.csv')
	latlong_df.head()

	# include this latitude value in the feature dataframe
	state_stats_df1 = (pd.merge(state_stats_df, latlong_df[['Latitude', 'State']],
	                           left_index=True, right_on='State').drop(columns=['State']))
	state_stats_df1.index = ordered_density_metric2state.values()

	st.write(state_stats_df1)

	st.write("The next feature is whether each Lower 48 state borders either the Atlantic or Pacific Ocean. This can potentially be an important feature because tourists and immigrants usually fly into the country in a coastal location")

	coastal_states = set('ME NH MA RI CT NY NJ PA MD DE VA NC SC GA FL WA OR CA'.split())
	state_stats_df1['is_coastal'] = [int(state in coastal_states) for state in state_stats_df.index]

	st.write(state_stats_df1)

	st.write("A potentially important state-level feature is the number of airline passengers arriving in the state. As we've seen with Covid-19, clusters have started in particular locations because visiters have come into these places with the virus from foreigns countries. The most readily available source for this data are the 'List of airports in [state]' Wikipedia article for each state. Each of these pages contains the number of commerical passenger boardings in 2016 for each airport in the state. Although commerical passenger arrivals are not included, it's reasonable to assume that the number of boardings and arrivals are closely related to each other. The values in the dictionary below represents the sum of the number of commerical passenger arrivals for the major airports in each state. Note: the number of major airports variesby state (e.g. the only major airport in Massachusetts in Logan, there are no major airports in Delaware, and there are three major airports in Kentucky (Cincinatti, Louisville and Lexington). Finally, the number of annual boardings in each state in normalized by the population of the given state, as this metric represents the relative influence of air traffic on the given state.")

	state2passengers = {'NY': 50868391, 
	                    'PA': 15285948 + 4670954 + 636916, 
	                    'NJ': 19923009 + 589091,
	                    'MD': 13371816,
	                    'IL': round((83245472 / 2) + (22027737 / 2)),
	                    'MA': 17759044,
	                    'VA': 11470854 + 10596942 + 1777648 + 1602631,
	                    'MO': 6793076 + 5391557 + 462126,
	                    'CA': (39636042 + 25707101 + 10340164 + 5934639 + 5321603 + 5217242 
	                           + 4969366 + 2104625 + 2077892 + 1386357 + 995801 + 761298),
	                    'MI': 16847135 + 1334979 + 398508,
	                    'CO': 28267394 + 657694,
	                    'MN': 18123844,
	                    'TX': 31283579 + 20062072 + 7554596 + 6285181 + 6095545 + 4179994 + 1414376,
	                    'RI': 1803000,
	                    'GA': 50501858 + 1056265,
	                    'OH': 4083476 + 3567864 + 1019922 + 685553,
	                    'CT': 2982194,
	                    'IN': 4216766 + 360369 + 329957 + 204352,
	                    'DE': 0,
	                    'KY': 3269979 + 1631494 + 638316,
	                    'FL': (20875813 + 20283541 + 14263270 + 9194994 + 4239261 + 3100624 + 2729129 
	                           + 1321675 + 986766 + 915672 + 589860),
	                    'NE': 2127387 + 162876,
	                    'UT': 11143738,
	                    'OR': 9071154,
	                    'TN': 6338517 + 2016089 + 887103,
	                    'LA': 5569705 + 364200,
	                    'OK': 1796473 + 1342315,
	                    'NC': 21511880 + 5401714 + 848261,
	                    'KS': 781944,
	                    'WA': 21887110 + 1570652,
	                    'WI': 3496724 + 1043185 + 348026 + 314909,
	                    'NH': 995403,
	                    'AL': 1304467 + 527801 + 288209 + 173210,
	                    'NM': 2341719,
	                    'IA': 1216357 + 547786,
	                    'AZ': 20896265 + 1594594 + 705731,
	                    'SC': 1811695 + 991276 + 944849 + 553658,
	                    'AR': 958824 + 673810,
	                    'WV': 213412,
	                    'ID': 1633507,
	                    'NV': 22833267 + 1771864,
	                    'ME': 886343 + 269013,
	                    'MS': 491464 + 305157,
	                    'VT': 593311,
	                    'SD': 510105 + 272537,
	                    'ND': 402976 + 273980 + 150634 + 132557 + 68829,
	                    'MT': 553245 + 423213 + 381582 + 247816 + 176730 + 103239,
	                    'WY': 342044 + 92805}

	# population of each state according to the 2010 census
	state2popn_2010 = {
	        'AL': 4779736,
	        'AR': 2915918,
	        'AZ': 6392017,
	        'CA': 37253956,
	        'CO': 5029196,
	        'CT': 3574097,
	        'DE': 897934,
	        'FL': 18801310,
	        'GA': 9687653,
	        'IA': 3046355,
	        'ID': 1567582,
	        'IL': 12830632,
	        'IN': 6483802,
	        'KS': 2853118,
	        'KY': 4339367,
	        'LA': 4533372,
	        'MA': 6547629,
	        'MD': 5773552,
	        'ME': 1328361,
	        'MI': 9883640,
	        'MN': 5303925,
	        'MO': 5988927,
	        'MS': 2967297,
	        'MT': 989415,
	        'NC': 9535483,
	        'ND': 672591,
	        'NE': 1826341,
	        'NH': 1316470,
	        'NJ': 8791894,
	        'NM': 2059179,
	        'NV': 2700551,
	        'NY': 19378102,
	        'OH': 11536504,
	        'OK': 3751351,
	        'OR': 3831074,
	        'PA': 12702379,
	        'RI': 1052567,
	        'SC': 4625364,
	        'SD': 814180,
	        'TN': 6346105,
	        'TX': 25145561,
	        'UT': 2763885,
	        'VA': 8001024,
	        'VT': 625741,
	        'WA': 6724540,
	        'WI': 5686986,
	        'WV': 1852994,
	        'WY': 563626
	}

	state_stats_df1['airport_boardings'] = [state2passengers[state] / state2popn_2010[state]
	                                        for state in state_stats_df.index]


	st.write(state_stats_df1)

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

	st.write("The next feature is the fraction of each state's population that falls into a set of age categories")

	age_df = pd.read_csv(pred_dir / 'age.csv')

	# merge age dataframe with dataframe that contains the rest of the features
	age_df['Location'] = [state2abbrev[state] for state in age_df.Location]
	state_stats_df2 = (pd.merge(state_stats_df1, age_df, left_index=True, right_on='Location')
	                  .drop(columns=['Location']))
	state_stats_df2.index = ordered_density_metric2state.values()

	st.write(state_stats_df2)

	st.write("The next feature is average temperature of each state during each of the four seasons of the year.")

	temps_df = pd.read_csv(pred_dir / 'temps.csv')

	temps_df['State'] = [state2abbrev[state] for state in temps_df.State]

	# merge temperature dataframe with dataframe that contains the rest of the features
	state_stats_df3 = (pd.merge(state_stats_df2, temps_df, left_index=True, right_on='State')
	                  .drop(columns=['State']))

	state_stats_df3.index = ordered_density_metric2state.values()

	st.write(state_stats_df3)

	st.write("It's possible that state-level political policies have an impact on the proliferation of virus infections. The Cook Partisan Voting Index taken from Wikipedia assigns a number to each state that indicates how strongly the state leads toward the Republican or Democratic Party based on recent state and federal elections. In our convention, a positive value signifies leaning Republican, while a negative value signifies leading Democratic.")

	state2partisan_score = {
	        'AL': 14,
	        'AR': 15,
	        'AZ': 5,
	        'CA': -12,
	        'CO': 1,
	        'CT': -6,
	        'DE': -6,
	        'FL': 2,
	        'GA': 5,
	        'IA': 3,
	        'ID': 19,
	        'IL': -7,
	        'IN': 9,
	        'KS': 13,
	        'KY': 15,
	        'LA': 11,
	        'MA': -12,
	        'MD': -12,
	        'ME': -3,
	        'MI': -1,
	        'MN': -1,
	        'MO': 9,
	        'MS': 9,
	        'MT': 11,
	        'NC': 3,
	        'ND': 17,
	        'NE': 14,
	        'NH': 0,
	        'NJ': -7,
	        'NM': -3,
	        'NV': -1,
	        'NY': -12,
	        'OH': 3,
	        'OK': 20,
	        'OR': -5,
	        'PA': 0,
	        'RI': -10,
	        'SC': 8,
	        'SD': 15,
	        'TN': 14,
	        'TX': 8,
	        'UT': 20,
	        'VA': -1,
	        'VT': -15,
	        'WA': -7,
	        'WI': 0,
	        'WV': 19,
	        'WY': 25
	}

	state_stats_df3['partisan_score'] = [state2partisan_score[state] for state in state_stats_df3.index]

	st.write(state_stats_df3)

	st.write("The following dataset was taken from a Stat139 problem set last semester and contains a range of socioeconomic, demographic and health indicators. These include:\n\n Cancer: prevalence of cancer per 100,000 individuals\n\n Hispanic: percent of adults that are hispanic \n\n Minority: percent of adults that are nonwhite\n\n Female: percent of adults that are female\n\n Income: median income\n\n Nodegree: percent of adults who have not completed high school\n\n Bachelor: percent of adults with a bachelor’s degree\n\nInactive: percent of adults who do not exercise in their leisure time\n\nObesity: percent of individuals with BMI > 30\n\n Cancer: prevalence of cancer per 100,000 individuals\n\n  We're not considering unemployment rate, as these rates are likely no longer accurate for many states.\n\nJust as with the density metric, the state-level value for each of these features is determined by calculating a weighted average of the measurements for each county, where the weights are the fraction of the state population that lives in the given county.")

	county_metrics_df = pd.read_csv(pred_dir / 'county_metrics.csv')


	county_metrics_df['state'] = [state2abbrev[state] for state in county_metrics_df.state]

	county_metrics_df = county_metrics_df[county_metrics_df.state.isin(lower_48)]

	st.write(county_metrics_df.head())

	state2pop_ = county_metrics_df.groupby('state').agg({'population': sum}).to_dict()['population']
	county_metrics_df['state_popn'] = [state2pop_[state] for state in county_metrics_df.state]

	metrics = ['hispanic', 'minority', 'female', 'unemployed', 'income', 'nodegree', 'bachelor', 'inactivity',
	          'obesity', 'cancer']

	for metric in metrics:
	    state2metric = (county_metrics_df.groupby('state').
	                    apply(lambda x: round((x['population'] * x[metric]) / x['state_popn'], 3))
	                    .groupby('state').sum()).to_dict()
	    
	    denom = 1000 if metric == 'income' else 1
	    state_stats_df3[metric] = [state2metric[state] / denom for state in state_stats_df3.index]

	st.write(state_stats_df3)

	st.write("The more people travel between states, the more closely related the states should be in terms of rate of virus infections. The Census Bureau Journey to Work datset reports the number of people that commute from any given county in the county to any other county in the country. This means we can aggregate these county to county commuting flows to determine the number of people that commute between any two states. From this data, we can create a symmetric matrix where the $i,j$ and $j,i$ elements represent the number of people that commute from state $i$ to state $j$ plus the number of people that commute from state $j$ to state $i$. However, just as with the number of annual boardings in each state, the final value of the number of people who commute between two states in normalized by the popualation of the given state. This means that this commuting matrix is no longer symmetric because the populations of state $i$ and state $j$ are different.")

	commuting_df_complete = pd.read_csv(pred_dir / 'commuting.csv')

	commuting_df = commuting_df_complete[['State Name', 'State Name.1', 'Workers in Commuting Flow']]

	commuting_df.rename(columns={'State Name': 'home_state', 
                             'State Name.1': 'work_state', 
                             'Workers in Commuting Flow': 'commuters'}, 
                   inplace=True)

	lower_48_full_name = [abbrev2state[abbrev] for abbrev in lower_48]
	commuting_df = commuting_df[commuting_df.work_state.isin(lower_48_full_name)]

	commuting_df['home_state'] = [state2abbrev[state] for state in commuting_df.home_state]
	commuting_df['work_state'] = [state2abbrev[state] for state in commuting_df.work_state]

	st.write(commuting_df.head(10))

	commuting_df['commuters'] = commuting_df['commuters'].apply(lambda x: int(''.join([y for y in x if y.isdigit()])))

	commuting_groupby_df = (commuting_df.groupby(['work_state', 'home_state'], as_index=False)
	                       .agg({'commuters': 'sum'}))

	# calculate the number of commuters between two states for all pairs of states
	for work_state in state_stats_df3.index:
	    vals = []
	    for home_state in state_stats_df3.index:
	        try:
	            num1 = int((commuting_groupby_df[(commuting_groupby_df.work_state == work_state)
	                       & (commuting_groupby_df.home_state == home_state)].commuters))
	            num2 = int((commuting_groupby_df[(commuting_groupby_df.work_state == home_state)
	                       & (commuting_groupby_df.home_state == work_state)].commuters))
	            num = num1 + num2
	            
	            num /= state2popn_2010[work_state]
	            
	        except TypeError:
	            num = 0

	        vals.append(num)

	    state_stats_df3[work_state + '_dest'] = vals

	st.write(state_stats_df3)

	st.write("States that are in close proximity may be similarly affected by viruses. Therefore, we include a column for each state in the design matrix that denotes whether that given states borders each of the other states.")

	# dictionary that maps each state in the Lower 48 to the states that directly border it or are not contiguous
	# but are very close (e.g. NJ and CT)
	state2neighbors = {'AL': {'AL', 'MS', 'TN', 'FL', 'GA', 'NC', 'SC'},
	                  'GA': {'GA', 'TN', 'FL', 'AL', 'SC', 'NC', 'MS'},
	                  'FL': {'FL', 'GA', 'AL', 'MS', 'SC'},
	                  'MS': {'MS', 'AL', 'TN', 'FL', 'LA', 'AR', 'GA'},
	                  'LA': {'LA', 'TX', 'AR', 'MS', 'OK', 'AL'},
	                  'SC': {'SC', 'FL', 'GA', 'NC', 'TN'},
	                  'NC': {'NC', 'SC', 'GA', 'TN', 'VA', 'KY'},
	                  'AR': {'AR', 'LA', 'TX', 'MS', 'TN', 'OK', 'MO', 'KY'},
	                  'VA': {'VA', 'NC', 'KY', 'WV', 'TN', 'DC', 'MD', 'DE'},
	                  'MD': {'MD', 'DC', 'VA', 'WV', 'DE', 'NJ', 'PA'},
	                  'DE': {'DE', 'MD', 'DC', 'NJ', 'PA'},
	                  'NJ': {'NJ', 'DE', 'MD', 'PA', 'NY', 'NJ', 'CT'},
	                  'NY': {'NY', 'NJ', 'PA', 'CT', 'MA', 'VT'},
	                  'CT': {'CT', 'NY', 'RI', 'MA', 'NJ'},
	                  'RI': {'RI', 'CT', 'MA'},
	                  'MA': {'MA', 'CT', 'RI', 'NH', 'VT', 'NY'},
	                  'NH': {'NH', 'VT', 'ME', 'MA'},
	                  'ME': {'ME', 'NH', 'MA', 'VT'},
	                  'VT': {'VT', 'NH', 'NY', 'MA'},
	                  'PA': {'PA', 'NY', 'NJ', 'MD', 'WV', 'OH', 'DE'},
	                  'WV': {'WV', 'DC', 'MD', 'PA', 'OH', 'KY', 'VA'},
	                  'OH': {'OH', 'PA', 'WV', 'MI', 'IN', 'KY'},
	                  'MI': {'MI', 'OH', 'WI', 'IN', 'IL'},
	                  'KY': {'KY', 'WV', 'OH', 'IN', 'IL', 'MO', 'TN', 'VA', 'AR', 'NC'},
	                  'TN': {'TN', 'KY', 'VA', 'NC', 'SC', 'GA', 'AL', 'MS', 'AR', 'MO', 'IL'},
	                  'IN': {'IN', 'KY', 'OH', 'MI', 'IL', 'WI'},
	                  'IL': {'IL', 'IN', 'MI', 'WI', 'IA', 'MO', 'KY', 'TN'},
	                  'WI': {'WI', 'IL', 'MN', 'MI', 'IA'},
	                  'MN': {'MN', 'MI', 'WI', 'IA', 'ND', 'SD', 'NE', 'IL'},
	                  'IA': {'IA', 'WI', 'MN', 'IL', 'MO', 'KS', 'NE', 'SD'},
	                  'MO': {'MO', 'IA', 'IL', 'KY', 'TN', 'AR', 'OK', 'KS', 'NE'},
	                  'ND': {'ND', 'SD', 'MN', 'MT', 'WY'},
	                  'SD': {'SD', 'ND', 'MN', 'IA', 'NE', 'MT', 'WY'},
	                  'NE': {'NE', 'SD', 'IA', 'MO', 'KS', 'WY', 'CO'},
	                  'KS': {'KS', 'NE', 'IA', 'MO', 'AR', 'OK', 'CO', 'TX', 'NM'},
	                  'OK': {'OK', 'KS', 'MO', 'AR', 'TX', 'NM', 'CO', 'LA'},
	                  'TX': {'TX', 'LA', 'AR', 'OK', 'NM', 'CO'},
	                  'MT': {'MT', 'ND', 'SD', 'WY', 'ID'},
	                  'WY': {'WY', 'MT', 'ND', 'SD', 'NE', 'CO', 'UT', 'ID'},
	                  'CO': {'CO', 'WY', 'NE', 'KS', 'OK', 'TX', 'NM', 'UT', 'AZ'},
	                  'NM': {'NM', 'CO', 'KS', 'OK', 'TX', 'AZ', 'UT'},
	                  'ID': {'ID', 'MT', 'WY', 'UT', 'NV', 'WA', 'OR'},
	                  'UT': {'UT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'NV'},
	                  'AZ': {'AZ', 'NM', 'CO', 'UT', 'NV', 'CA'},
	                  'WA': {'WA', 'ID', 'OR'},
	                  'OR': {'OR', 'WA', 'ID', 'NV', 'CA'},
	                  'NV': {'NV', 'ID', 'OR', 'UT', 'AZ', 'CA'},
	                  'CA': {'CA', 'OR', 'NV', 'AZ'}
	                 }

	     
	for neighboring_state in state_stats_df3.index:
	    states = [int(neighboring_state in state2neighbors[state]) for state in state_stats_df3.index]
	    state_stats_df3[neighboring_state + '_is_neighbor'] = states  
	
	st.write(state_stats_df3)

	st.write("The proportion of each state that is vaccinated may affect the number of people who are infected with the flu. Therefore, we include information on the adult and child vaccination rate for each state.")
	flu_df = pd.read_csv(pred_dir / 'flu.csv')
	flu_df['State'] = [state2abbrev[state] for state in flu_df.State]

	state_stats_df4 = (pd.merge(state_stats_df3, flu_df, left_index=True, right_on='State').drop(columns=['State']))
	state_stats_df4.index = state_stats_df3.index

	st.write(state_stats_df4)

	st.write("Smoking may also affect suspectibility to viruses such as the flu and Covid-19, so we include a feature that reports the fraction of adults who smoke in each state.")

	state2smoking_rate = {
        'AL': 20.9,
        'AR': 22.3,
        'AZ': 15.6,
        'CA': 11.3,
        'CO': 14.6,
        'CT': 12.7,
        'DE': 17.0,
        'FL': 16.1,
        'GA': 17.5,
        'IA': 17.1,
        'ID': 14.3,
        'IL': 15.5,
        'IN': 21.8,
        'KS': 17.4,
        'KY': 24.6,
        'LA': 23.1,
        'MA': 13.7,
        'MD': 13.8,
        'ME': 17.3,
        'MI': 19.3,
        'MN': 14.5,
        'MO': 20.8,
        'MS': 22.2,
        'MT': 17.2,
        'NC': 17.2,
        'ND': 18.3,
        'NE': 15.4,
        'NH': 15.7,
        'NJ': 13.7,
        'NM': 17.5,
        'NV': 17.6,
        'NY': 14.1,
        'OH': 21.1,
        'OK': 20.1,
        'OR': 16.1,
        'PA': 18.7,
        'RI': 14.9,
        'SC': 18.8,
        'SD': 19.3,
        'TN': 22.6,
        'TX': 15.7,
        'UT': 8.9,
        'VA': 16.4,
        'VT': 15.8,
        'WA': 13.5,
        'WI': 16,
        'WV': 26,
        'WY': 18.7
	}

	state_stats_df4['smoking_rate'] = [state2smoking_rate[state] / 100 for state in state_stats_df4.index]

	st.write(state_stats_df4)

	st.write("## Bayesian Model")

	st.write("### Motivation")

	st.write("Before describing the model, it's important to first discuss the motivation behind it in the first place. The wILI time series clearly show that the states are affected differently by the flu. Therefore, we wanted to determine whether there are any state-level features that account for the disrepencies between the states. If we could identify these particular features, then we'd also be able to figure out which states are intrinsically linked based on their attributes.") 
	st.write("This information would then allow us to transfer this knowledge about the flu to Covid-19. Because both the flu and Covid are viruses, we'd expect some of the underlying risk factors of flu to generalize to Covid as well. We could then take one of two routes: first, we could assess if the interstate correlations discovered from the flu data apply in the case of Covid by comparing the number of Covid cases among different states. And second, we could assume that the flu relationships apply in the case of Covid and use these insights to look deeper than just the raw Covid numbers. For example, if the flu analysis reveals that two states share many similar characteristics, and one of these states has more Covid cases per 1000 people but also has more testing, then we may believe that the second state has more case of Covid than are reported. Alternatively, we can identify states that, based on their characteristics (e.g. high density, high obesity rate), are more susceptible to a major spike in Covid cases and thus should take additional precautions when opening up their states.")

	st.write("### Model Formulation")

	st.write("If the state wILI rates are correlated with each other, then we should, in theory, be able to predict the wILI rate in a given state and for a given week from the wILI rates of all the other states for the same week. Because correlated states may have similar flu trajectories but have different raw wILI rates, it's more robust to predict the weekly percent change in wILI rather than the absolute change in wILI. This means that we want to predict the trend in the number of flu cases for each state based on the trends of all the other states at the same time.")

	st.write("The big question is obviously how to use the percent change in the wILI rate of every other state to predict the percent change in the wILI rate for a single state. Because some states are more closely correlated with a given state than others, it makes sense to predict the percent change for a given state to be a weighted average of the percent changes of the other weeks, where the weights should ideally be proportional to the underlying correlation between the two states. For example, if we were trying to predict the trend in New York, we'd take into account the trend of every other state (except for Alaska and Hawaii), but the influence of each of these states on our overall prediction for New York would vary (e.g. the influence of New Jersey and Connecticut may be high, while the influenced of Idaho and Nebraska may be low).")

	st.write("Converting this into formal notation, let's define $\\delta_i$ to be the percent change in the wILI rate between two consecutive weeks for state $i$, and define $\\alpha_{ij}$ to be the weight coefficient of state $j$ on state $i$. We predict each $\\delta_i$ as:")

	st.latex("\\delta_i \\sim N\\left(\\frac{\\sum_{j=1}^{48}\\alpha_{ij}\\delta_jI(j \\neq i)}{\\sum_{j=1}^{48}\\alpha_{ij}I(j \\neq i)}, {\\sigma_{i}}^2\\right)")

	st.write("where ${\\sigma_{i}}^2$ is a state-specific variance. Intuitively, the lower the value of ${\\sigma}^2$ for a given state, the more the variation in the state's wILI trend can be explained by the wILI trends of the other states, and vice versa.")

	st.write("Next, we want to link the $\\alpha_{ij}$ weights to the features associated with each state such that states with more similar characteristics and high rates of interstate travel have higher $\\alpha_{ij}$ and $\\alpha_{ji}$ values and vice versa. Additionally, we only want a few of the $\\alpha_{ij}$s corresponding to state $i$ to be large, and the rest to be small (in a similar spirit to regularization). We can accomplish both of these features as follows: first, each $\\alpha_{ij}$ is modelled as being distributed according to an exponential distribution with a scale (i.e. inverse rate) parameter of $\\lambda_{ij}$. Because an exponential distribution is right skewed and has most of its mass near zero, this ensures that most of the $\\alpha_{ij}$ that are drawn from exponential distributions will take on relatively small values, while only a few will take on relatively large values. Next, we link the scale parameter ($\\lambda_{ij}$) of this exponential distribution to the state-level features by setting the log of $\\lambda_{ij}$ equal to the linear predictor function (taking the log is necessary to map the domain of the scale parameter (all positive real numbers) to the domain of the linear prediction function (the entire real line)).")

	st.write("Translating this into formal notation:")

	st.latex("\\alpha_{ij} \\sim Expo(\\lambda_{ij})")

	st.latex("log(\\lambda_{ij}) = \\beta_0 + \\beta_1X_1 + ... + \\beta_kX_k")

	st.write("In this case the linear predictor function is a little different that usual. Two of the predictors (normalized number of commuters between states $i$ and $j$ and the indicator of whether state $j$ borders state $i$) are included in the usual form of $\\beta_iX_i$, where a unit increase in $X_i$ corresponds to a $\\beta_i$ increase in the linear predictor. However, the rest of the predictors are state-level features such as obesity rate and density. This means that we don't care about the raw values of these features; instead, we only care about the difference between the values for state $i$ and state $j$. Therefore, each of the predictors is defined to be $|X_i - X_j|$, such that the predictor value is 0 when the two states have the same feature value, and increases as the difference between the two states grows.")

	st.write("Finally, because this is a Bayesian model, we need to define a prior distribution for the model parameters, which in this case are the $\\beta$ coefficient associated with each predictor variable and the ${\\sigma}^2$ parameter associated with each state. Because we have no substantial prior domain knowledge, we placed relatively uninformative priors on these parameters. Putting all of these components together produces the following generative model:")

	st.latex("\\delta_i \\sim N\\left(\\frac{\\sum_{j=1}^{48}\\alpha_{ij}\\delta_jI(j \\neq i)}{\\sum_{j=1}^{48}\\alpha_{ij}I(j \\neq i)}, {\\sigma_{i}}^2\\right)")

	st.latex("\\sigma_{i}^{2} \\sim Inv-Gamma(2, 2)")

	st.latex("\\alpha_{ij} \\sim Expo(\\lambda_{ij})")

	st.latex("log(\\lambda_{ij}) = \\beta_0 + \\beta_1X_1 + ... + \\beta_kX_k")

	st.latex("\\beta_i \\sim N(0, 5^2) ")

	st.write("Performing inference for this model yields the posterior distribution of the $\\beta$s and the ${\\sigma}^2$, but we only really care about the $\\beta$s. Because the exponential distribution is parameterized by a scale parameter rather than the usual rate parameter, the expected value of the distribution is equal to the scale parameter. This means that a larger $\\lambda_{ij}$ value corresponds, on average, to a higher $\\alpha_{ij}$ coefficient, and because the linear predictor function is defined to be the log of $\\lambda_{ij}$, this in turn means that a larger linear predictor corresponds, on average, to a higher $\\alpha_{ij}$ coefficient. For the two predictors that are not differences between the two given states, this means that a positive $\\beta$ coefficent indicates that a unit increase in the predictor value produces a stronger correlation between the two given states and vice versa. On the other hand, for the rest of the predictors that are included as differences between certain features of the two states, a strong correlation between two given states is signified by a negative $\\beta$ coefficient. This is the case because the predictor value represents the absolute differences between the features of the states, so a larger predictor value corresponds to a larger discrepancy between the states. Thus, the corresponding $\\beta$ coefficient can be interpreted as a penalty parameter, such that states that are less similar in terms of the given feature are less correlated with each other (assuming the $\\beta$ coefficient value is negative).")

	st.write("Overall, the model provides us with two interpretative results. First, the $\\beta$ coefficients indicate which features contribute to the correlation between the wILI time series of different states. And second, the $\\beta$ coefficients tell us about the $\\alpha_{ij}$ weights, which, in turn, inform us about which states are highly correlated with each other based on the fundamental characteristics of the states.")

	st.write("Finally, one major advantage of this model is that the observations (i.e. the percent change in the wILI rate for a given week) are independent of each other conditioned on the percent changes of the other states for the same week. This means that unlike in a classic time seris model, the past wILI rates of a state are irrelevant to predicting the percent change in the wILI rate at any given time. This greatly simplifies things, as it's much easier to deal with independent observations than it is to handle observations that are correlated with previous observations.")

	predictor_df = pd.read_csv(pred_dir / 'state_stats.csv')
	predictor_df.drop(index='FL', inplace=True, errors='ignore')
	flu_percent_change_df = pd.read_csv(pred_dir / 'flu_percent_change_imputed_48.csv')
	week_nums = flu_percent_change_df.week_num
	flu_percent_change_df.drop(columns='week_num', inplace=True)

	flu_percent_change_df = flu_percent_change_df[predictor_df.index]

	st.write("Weekly percent change in wILI rate by state:")
	st.write(flu_percent_change_df.head())

	# predictors that are compared between states
	comparison_predictors = ['density_metric', 'Latitude', 'is_coastal', 'airport_boardings', 'Children 0-18', 
	                          'Adults 19-25', 'Adults 26-34', 'Adults 35-54', 'Adults 55-64', '65+', 
	                         'partisan_score', 'hispanic', 'minority', 'female', 
	                         'income', 'nodegree', 'bachelor', 'inactivity', 'obesity', 'cancer',
	                         'overall_vacc_rate', 'child_vacc_rate', 'smoking_rate']
	season_predictors = ['spring', 'fall', 'winter']

	# predictors that are not compared between states
	no_comparison_predictors = ['commuters', 'is_neighbor']

	st.write("An important preprocessing step is to standardize each of the predictors (except for `is_coastal` and `is_neighbor` as these variables only take on the values 0 and 1. This ensures that the $\\beta$ coefficients associated with each predictor are all on the same scale and thus are easily comparable to each other. Additionally, ensuring the the $\\beta$ parameters lie in a similar range may help with the MCMC sampling.")


	predictors_to_standardize = [x for x in comparison_predictors if x != 'is_coastal'] + season_predictors

	# there are no observations during the summer so we don't need the summer weather predictor
	predictor_df_standardized = predictor_df.drop(columns='summer')
	for predictor in predictors_to_standardize:
	    data = predictor_df_standardized[predictor]
	    mean = np.mean(data)
	    std = np.std(data)
	    predictor_df_standardized[predictor] = [(x - mean) / std for x in data]

	commute_columns = [column for column in predictor_df_standardized if column.endswith('_dest')]
	commute_vals = predictor_df_standardized[commute_columns].to_numpy().flatten()
	commute_mean = np.mean(commute_vals)
	commute_std = np.std(commute_vals)

	for commute_column in commute_columns:
	    predictor_df_standardized[commute_column] = [(x - commute_mean) / commute_std 
	                                                 for x in predictor_df_standardized[commute_column]]
	    
	comparison_preds_df = predictor_df_standardized[comparison_predictors + season_predictors]

	st.write("Resulting state feature dataframe:")
	st.write(predictor_df_standardized)

	# determine season from week of the year
	def get_season(week):
	    if week >= 52 or week < 13:
	        return np.array([0, 0, 1])
	    if 13 <= week < 26:
	        return np.array([1, 0, 0])
	    if 39 <= week < 52:
	        return np.array([0, 1, 0])
	    raise 


	predictor_num = len(comparison_predictors) + len(season_predictors) + len(no_comparison_predictors)
	state_num = flu_percent_change_df.shape[1]
	comparison_preds_num = len(comparison_predictors)
	obs_num = len(flu_percent_change_df)

	# indicate which season each observation fall into 
	season_indictor_array = np.zeros((obs_num, state_num - 1, len(season_predictors)))
	for i, week_num in enumerate(week_nums[1:]):
	    season_indictor_array[i, :, :] = np.repeat(get_season(week_num)[np.newaxis, :], state_num - 1, axis=0)

	st.write("`Y_target` is a 1D array that contains the percent change of each state for each week of the time series that is included in the analysis. This is the variable we want to predict for each observation. Because there are 47 states (Lower 48 except for Florida) and 217 observations for each state, this array has a length of $47*217=10199$. \n\n`Y_state_idx` is a 1D array of the same length as `Y_target` that represents the specific state associated with each `Y_target` value. Therefore, it takes on values between 0 and 46. This is necessary to pick out the variance parameter corresponding to the given state. \n\n`X` is a 3D design matrix. The first axis has a length equal to the total number of observations (10199). The second axis has a length of 46, which represents the $47-1=46$ other states from which we're trying to predict the final state. And the first axis has a length of 29, which contain the 28 predictors in addition to an intercept term, which is simply the value of 1. Therefore, this `X` matrix contains all the predictors for each state for each observation.\n\n`X_flu` is a 2D array. The first axis has a length equal to the total number of observations (10199), while the second axis has a length of 46 and represents the percent change in wILI rate for all the $47-1=46$ other states from which we're trying to predict the final state. Therefore, this array is contains all the $\\delta_jI(j \\neq i)$ values for each observation.")

	Y_target = np.zeros(state_num * obs_num)
	X = np.zeros((Y_target.shape[0], state_num - 1, predictor_num + 1))
	Y_state_idx = np.zeros(Y_target.shape[0], dtype=int)
	X_flu = np.zeros((Y_target.shape[0], state_num - 1))
	X.shape

	for idx, state in enumerate(predictor_df_standardized.index):
    
	    # response variable
	    Y_target[obs_num * idx: obs_num * idx + obs_num] = flu_percent_change_df[state]
	    
	    # percent change of other states
	    X_flu[obs_num * idx: obs_num * idx + obs_num, :] = flu_percent_change_df.drop(columns=state).to_numpy()
	    
	    # index of response state
	    Y_state_idx[obs_num * idx: obs_num * idx + obs_num] = [idx] * obs_num
	    
	    state_comparison_preds = np.array(comparison_preds_df.loc[state])
	    
	    constant_design_matrix = np.zeros((X.shape[1], X.shape[2]))
	    constant_design_matrix[:, 0] = np.ones(state_num - 1)
	    
	    # two predictors that aren't differences between states: neighboring state and number of commuters
	    other_states_preds_df = predictor_df_standardized.drop(index=state)
	    not_difference_matrix = other_states_preds_df[[state + '_is_neighbor', state + '_dest']].to_numpy()
	    constant_design_matrix[:, 1: 1 + len(no_comparison_predictors)] = not_difference_matrix
	    
	    # the rest of the predictors are differences between two states
	    other_states_comparison_preds_array = comparison_preds_df.drop(index=state).to_numpy()
	    difference_matrix = abs((other_states_comparison_preds_array - state_comparison_preds) ** 1)
	    constant_design_matrix[:, 1 + len(no_comparison_predictors):] = difference_matrix
	    
	    constant_design_matrix_3D = np.repeat(constant_design_matrix[np.newaxis, :, :], repeats=obs_num, axis=0)
	    
	    # pick out appropriate season and set the rest of the temperature predictors to zero
	    constant_design_matrix_3D[:, :, -len(season_predictors):] *= season_indictor_array 
	    
	    X[obs_num * idx: obs_num * idx + obs_num, :, :] = constant_design_matrix_3D 
	
	st.write("The observations are shuffled before they are inputted to the pymc3 model.")

    # randomly shuffle the observations 
	np.random.seed(109)
	indices = np.arange(len(Y_target))
	np.random.shuffle(indices)
	Y_target_random = Y_target[indices]
	X_flu_random = X_flu[indices]
	X_random = X[indices]
	Y_state_idx_random = Y_state_idx[indices]

	st.write("See bottom of document for model specification.")

	st.write("Just as we did in HW3, it's important to first check whether the generative model is correctly specified. This can be done by hardcoding the  values for the parameters, generating response variables from these parameters and then trying to infer the parameters using MCMC.")

	st.write("The sampling took a whopping 13 hours to sample just 500 times for each chain (with a 500 burn-in sample). However, as shown below the results confirm that the model was correctly specified, as the majority of the true $\\beta$ values lie within the corresponding 94 percent credible interval. Therefore, performance inference for the actual data should yield reliable results.\n\nHowever, carrying out inference on this synthetic data reveals several issues. First, many of the r_hat values are significantly larger than 1.0, which means that more than 500 samples are needed for the chains to converge to the posterior distribution. And second, the fact that the sampling took so long may indicate that the uninformative priors are too flat and make it difficult for the NUTS sampler to sample points from the true posterior distribution. To address these issues, the number of samples is increased from 500 to 1000 and a semi-informative prior is placed on the $\\beta$ and $\\sigma^2$ parameters ($N(0, 25)$ for each of the $\\beta$s and $Inv-Gamma(2, 2)$ for each $\\sigma^2$.")

	#sim_trace_df = pd.read_csv(pred_dir / 'sim_trace.csv')
	#st.write(sim_trace_df)

	st.write("Unfortunately I ran into major issues running MCMC for the actual data. A burn-in of 500 and a sample of 1000 should have taken around 18 hours to finish. However, the first time I ran it, it was 80 percent complete after 14 hours and then my screen saver didn't turn off and the notebook shut down. I then tried running in a second time, and this time it again was 80 percent done after another 14 hours and then encountered a memory failure issue that terminated the notebook. Therefore, the third time I only asked for 500 samples, even though I knew this likely wouldn't be large enough for the sampler to converge. It took 14 hours to run but finished successfully. Even so, the model was so unwieldy that it took an additional two hours just to save the model and create a summary dataframe.")

	st.write("Results of MCMC sampling:")

	trace_df = pd.read_csv(pred_dir / 'trace.csv')
	st.write(trace_df)

	st.write("Unfortunately, most of the r_hat values of the $\\beta$ coefficients are extremely inflated (the average r_hat value is just under 2.0). This means that the sampler hasn't come close to converging and means that it's pointless to try to interpret the sign or the magnitude of the coefficients. At this point, we ran out of time. However, if we had more time, we'd randomly select a subset of the observations and get more samples for these observations, as it's better to have trustworthy results on less data than it is to have unreliable results on the entire datset.")

	st.write("While the results of the inference were unreliable, it's still worthwhile to discuss what the next steps would have been in the analysis. First, we would check the sign and 94 percent credible interval of each of the $\\beta$ coefficients to see if the majority of them make intuitive sense (i.e. negative coefficients for the difference predictors and positive coefficients for the non-difference predictors.) Next, we would evaluate the predictive power of the model and test the model assumptions at the same time. This could be done by first calculating the predictive power of a baseline naive model where the average of all the other states is used to predict for the percent change in the final state (in other words, where the weights associated with each state are the same). Because the likelihood function is modelled as a normal distribution, the optimal loss function is the mean squared error. The predictions would be performed for each state separately. \n\nAfter calculating the MSE for the naive model, we'd evaluate the Bayesian model as follows: first, we'd sample hundreds of times from the posterior distribution of each of the $\\beta$ coefficients. Then, for each sample, we'd work our way up the model (i.e. sample an $\\alpha$ for each state) and calculate the mean of the prediction. We'd then plot the residuals by subtracting the predicted percent change from the true percent change. Calculating the average of the square of the residuals would give us the MSE, which we'd compare to the baseline model to see if this model has any increased predictive power. Meanwhile, we'd plot these residuals to assess the assumption that the observations are normally distributed about the weighted average of the percent change of each of the other states. If this is the case, then we'd expect the distribution to being normally distributed around 0.0. Finally, we can calculate the variance of the residuals for each state and compare this sample variance to the posterior distribution of $\\sigma^2$ for each state to check if they are consistent with each other.")

	st.write("Model specification in pymc3:")

	with st.echo():
		model = pm.Model()

		with model:
		    # define prior distribution for beta parameters 
		    beta = pm.Normal('beta', mu=0, sigma=5, shape=predictor_num + 1)
		    
		    # define prior distribution for state-specific variance parameter
		    sigma_sq = pm.InverseGamma('sigma_sq', alpha=2, beta=2, shape=state_num)
		    
		    # calculate the linear predictor for each state by multipling the 3D X design matrix with the vector
		    # of beta parameters
		    nu = pm.Deterministic('nu', pm.math.dot(X_random, beta))
		    
		    # calculate the lambda parameter for each state by exponentiating the linear predictor
		    lambda_ = pm.Deterministic('lambda', pm.math.exp(nu))
		    
		    # sample an alpha random variable for each state from an exponential distribution with the 
		    # corresponding rate parameter
		    alpha = pm.Exponential('alpha', lam=1/lambda_, shape=(X_random.shape[0], state_num - 1))
		    
		    # calculate the mean of each response variable by taking the dot product between the alpha vector
		    # and the vector of the percent change in the wILI rates of the other 46 states and dividing by the 
		    # sum of the alpha weights
		    mu = pm.Deterministic('mu', pm.math.sum(alpha * X_flu_random, axis=1) / pm.math.sum(alpha, axis=1))
		    
		    # define the response variable to be normally distributed about the mean and with a standard deviation that
		    # is the square root of the variance parameter associated with the given state
		    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=pm.math.sqrt(sigma_sq[Y_state_idx_random]), observed=Y_target_random)


