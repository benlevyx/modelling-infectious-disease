import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from PIL import Image

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import csv

plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config

def main():
    st.title('Covid Forecasting: CAR')
    st.write('In this section, we will explore how we can gain knowledge from 10 years of flu data to '
             'better predict the spread of Covid-19 over the US states. More specifally, we build the a version of'
             ' a so-called Conditional Autoregressive (CAR) Model. This Streamlit page will walk through the main '
             'ideas of the model, for more details we refer to the notebook: '
             'https://github.com/benlevyx/modelling-infectious-disease/blob/master/notebooks/CAR%20Model%20Flu.ipynb')
    st.subheader('1. Get the data')
    st.write('First, we should have a look at the data. For the Covid-19 data, we found that the New York Times '
             'released a publicly available dataset (source:https://data.humdata.org/dataset/nyt-covid-19-data). It contains the daily number of recorded cases per US '
             'state since January 21 2020, which was the date of the first confirmed case in the US.')\

    def fetch_and_clean_data():
        flu_dir = config.data / 'raw'
        filename = 'Covid_data.csv'

        df = pd.read_csv(flu_dir / filename)
        return df

    covid_data = fetch_and_clean_data()
    covid_data = covid_data.fillna(0.0)

    st.write(covid_data)

    fig = plt.figure(figsize=(20, 10))
    for state_name in covid_data.columns:
        if state_name != 'date' and state_name != 'Unnamed: 0':
            plt.plot(range(len(covid_data)), covid_data[state_name])

    plt.xlabel('Day since Jan 21 2020', fontsize = 25)
    plt.ylabel('Cases', fontsize = 25)
    plt.title('Covid-19 cases for all US states over time', fontsize = 25)
    st.pyplot(fig)

    st.subheader('2. Conditional Autoregressive (CAR) model ')
    st.subheader('2.1 CAR in general and application to the time-dependent spread of deseases ')
    st.markdown('We will first discuss the conditional autoregressive (CAR) model in general. '
             'It is widely used to model the spatial variation of the response variable $y_i$, '
             'where it is assumed that the probability of values estimated for a variable $z_i$ are'
             ' conditional on neighboring values $z_j$. As such, it is a natural way to study the '
             'spatial relations present in specific data. It is thus potentially interesting to apply'
             ' a CAR model to our geographically spread data on deseases. \n '
             'Consider a general Spatial Regression Model (SAR) to start with: \n')
    st.latex('y_i = \\beta X_i  + z_i + \epsilon_i')
    st.markdown('Where:\n'
             '- $y_i$ is the response variable at node i, \n - $X_i$ the predictor variables measured at the same node i as $y_i$, \n,'
             '- $\\beta$ the local regression coefficients, \n'
             '- $z_i$ a latent spatial random error $z_i \sim N(0, \Sigma_i)$, \n'
             '- $\epsilon_i$ an independent error $\epsilon_i \sim N(0, \sigma_{\epsilon_i}^2)$ \n')
    st.markdown('\n In the Conditional AR model, the $z_i$ variable depends on the neighbouring values $z_j$ '
             'for i distinct from j \n'
             '$$z_i | z_j, i \\neq j \sim N(\sum_{j \\neq i} c_{ij} z_j, m_{ii})$$ \n'
             'Where $c_{ii} = 0$. As such, there are three main contributions to the response variable $y_i$:'
             ' a regression of locally measured predictors, a conditional spatial term and a random error '
             'that is specific to the location. The matrix $C$ is often developed as $\\rho W$ where $W$ is '
             'the neighborhood matrix and $\\rho$ an autocorelation factor. Matrix $W$ contains the underlying '
             'relationship between the nodes present in the problem in the form of values $w_{ij}$ that is a '
             'proxy for the weighted spatial impact of node j on node i \n'
             '(Reference to: https://eprints.qut.edu.au/115891/1/115891.pdf). \n')
    st.markdown('The CAR model sounds like a very interesting idea to apply to a time-series of spread of '
             'deseases per state. First, it is natural that each state corresponds to 1 node i. Next, we can '
             'easily embed the temporal factor in a similar way to regular AutoRegression for time series, '
             'where the previous observations serve as local predictors. The spatial factor then comes in as '
             'before. More specifically: \n')
    st.latex('Y_{ti} = \\beta X_i + z_i + \epsilon_i')
    st.markdown('Where: \n'
                '- $Y_{ti}$ is for instance the Wili rate in state i at time t. We still want to '
             'predict the response variable over time at a specific location, so this remains the response '
             'variable. \n - $X_i$ is a vector containing all the local predictors. In the case of a AR time'
             'series model, this will contain the N previous Wili observations in time in state i.\n'
             '- $\\beta$ is a vector containing the local AR time-lag-regression coefficients. \n'
             '- $z_i$ is the latent spatial random error, which will be conditional on the neighbouring states,'
             'where neighbour still needs to be defined. \n - $\epsilon_i$ an independent error'
             '$\epsilon_i \sim N(0, \sigma_{\epsilon_i}^2)$ \n '
                ''
             'The spatial factors can then still be determined'
             ' in the same way: \n'
             '$$z_i | z_j , i \\neq j \sim N(\sum_{j} c_{ij} z_j, m_{ii})$$')
    st.markdown('Following standard practice, we can define the matrix $C = \\rho W$. Now it comes down to '
             'come up with a reasonable value for the correlation variable $\\rho$ and the construction of '
             'neighbor matrix $W$.')

    st.markdown('The initial idea was to apply the CAR model to the flu, where the weights of the neighbour '
                'matrix W would be drawn from a distribution that depends on the state statistics. '
                'However, we realized that: \n'
                '- this would be tricky to fit using PyMC3 and that \n'
                '- this is not the goal of CAR models, **as the neighbor characteristics should be '
                'intrinsic to the problem.**\n')
    st.markdown('As such, it would make more sense to use the model with clever, fixed values for the '
                'coefficients in W. This is where the flu data can leveraged in modeling Covid!'
                '**We will use the 10 years of Wili data in all states to come up with a reasonable'
                ' construction of W, capturing the definition of neighbor for all states in the context of '
                'the spread of a desease. We will then be able to use this fixed W to model state-specific'
                ' spread of Covid-19 for all US states.** ')

    st.subheader('2.2 Building the model')
    st.markdown('There will be two big steps here: \n'
                '- Construct W using the flu data.\n'
                '- Build the CAR model for Covid using this W \n')
    st.markdown('Nice resource: https://docs.pymc.io/notebooks/PyMC3_tips_and_heuristic.html')

    st.markdown('W will first construct W using the flu data. How does one capture the similarity between '
                'different states for a set of time series? The first thing that comes to mind is correlation!'
                ' Let\'s start by constructing W with: \n')
    st.latex('w_{ij} = \\rho_{ij}')
    st.markdown('Where $\\rho_{ij}$ corresponds to the correlation of the 10 year time series of state i '
                'with the 10 year time series of state j. ')

    flu_data = pd.read_csv(config.data / 'flu_ground_truth_imputed.csv')
    flu_data = flu_data.reindex(sorted(flu_data.columns), axis=1)
    corr_df = flu_data.corr()
    W_0 = corr_df.values
    st.write(corr_df)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(W_0)
    plt.title('Pixelmap of W using correlation', fontsize = 25)
    st.pyplot(fig)

    st.markdown('Although the correlation would make sense as weight between states, we would miss out on an '
                'important aspect of the definition of neighbor, being that a state can only have a limited '
                'amount of neighbors. So ideally, we wish to make W a sparse matrix, where only the '
                'coefficients corresponding to the highest correlation between states i and j are non-zero.\n'
                'Driving the least important coefficients to exactly zero sounds like LASSO! '
                '**So let\'s build a plain vanilla linear regression model for each state\'s time series, '
                'using all the other states as predictors. We\'ll fit the model using a strong '
                '$l_1$-regularization, which will lead to only a limited amount of non-zero coefficients.** '
                'Note that the Lasso Sklearn implementation has an attribute \'positive\' which makes sure '
                'that the model only considers positive values for the coefficients. ')

    @st.cache()
    def get_w_lasso():
        lamb = 0.01
        N = len(flu_data.columns)
        W = np.zeros((N,N))

        for i, state_name in enumerate(flu_data.columns):
            y = flu_data[state_name].values
            df_to_work_with = flu_data.drop([state_name], axis = 1)
            X = df_to_work_with.values
            lasso_obj = Lasso(alpha = lamb, positive = True)
            lasso_obj.fit(X,y)
            coef = lasso_obj.coef_
            left = coef[:i]
            right = coef[i:]
            W[i,:i] = left
            W[i, i+1:] = right
        return W
    W = get_w_lasso()
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(W)
    plt.title('Pixelmap of W using LASSO', fontsize = 25)
    st.pyplot(fig)

    st.markdown('Let\'s have a look whether these results make sense. ')

    def print_lasso_states():
        state_name = 'Arizona'
        i = 1
        neighbors = []
        coefs = []
        for k, coef in enumerate(W[i,:]):
            if coef > 0:
                coefs.append(coef)
                neighbors.append(flu_data.columns[k])
        n_neigh = len(neighbors)
        max_coefs = np.array(coefs).argsort()[::-1][1:n_neigh+1]
        neighbors = np.array(neighbors)[max_coefs]
        st.markdown('For state {}, the identified neighbours are: '.format(state_name))
        st.write(list(neighbors))
        corr_list = W_0[i, :].copy()
        max_coefs = corr_list.argsort()[::-1][1:n_neigh+1]
        max_corr_states = flu_data.columns[max_coefs]
        st.markdown('The max correlation states were: ')
        st.write(list(max_corr_states))

    print_lasso_states()

    st.markdown('This seems to make sense! So let\'s continue using these coefficients.')

    covid_data = covid_data.drop(['Unnamed: 0', 'date', 'Alaska','District of Columbia', 'Hawaii',
                                  'Guam', 'Florida', 'Northern Mariana Islands',
                                  'Virgin Islands', 'Puerto Rico'], axis = 1)

    st.markdown('Using this, we can now start building the CAR model. We will first do so by predicting '
                'one day ahead using one day lag. The model can then be implemented with an MCMC using PyMC3. '
                'For details on the implementation, it is recommended to have a look at the notebook on Github.'
                ' Below you can view the one day ahead predictions per state.')

    CAR_model_lag1 = pd.read_csv(config.data / 'processed' / 'CARModelLag1.csv')
    CAR_model_lag1.set_index('Unnamed: 0', inplace = True)

    params_dict = dict()

    for k, state_name in enumerate(covid_data.columns):
        params_k = [CAR_model_lag1['mean']['beta0[{}]'.format(k)],
                    CAR_model_lag1['mean']['betas[{}]'.format(k)],
                    CAR_model_lag1['mean']['phi[{}]'.format(k)]]
        params_dict[state_name] = params_k

    state = st.selectbox('Select state for which you wish to see the CAR predictions. It may take a second.',
		 covid_data.columns)


    def lag1_pred(state):
        # predict
        pred_state = []
        for i, day in enumerate(range(start + lag, len(covid_data) - pred_ahead)):
            params = params_dict[state]
            lag_X = covid_data[state][day]/covid_data[state][day-1]
            log_pred = params[0] + params[1]*lag_X + params[2]
            pred = np.exp(log_pred)
            T_pred = pred*covid_data[state][day]
            pred_state.append(T_pred)

        rmse = np.sqrt(mean_squared_error(pred_state,
                                 covid_data[state].iloc[start + lag: len(covid_data) - pred_ahead]))
        rmse = np.round(rmse, 3)

        return pred_state, rmse

    # plot
    pred_ahead = 1
    lag = 1
    start = 56
    pred_state, rmse = lag1_pred(state)
    fig2 = plt.figure(figsize = (12,8))
    plt.plot(range(start, len(covid_data[state])), covid_data[state][start:], label = 'Real data')
    plt.plot(range(start+2, len(covid_data[state])),
             pred_state, label = '1 lag CAR prediction - RMSE = {}'.format(rmse))
    plt.title('CAR predictions of Covid cases for {}'.format(state), fontsize = 20)
    plt.xlabel('Day since Jan 21 2020', fontsize = 20)
    plt.ylabel('# cases', fontsize = 20)
    plt.legend(fontsize = 20)
    st.pyplot(fig2)

    st.markdown('Overall, the one day ahead predictions of the CAR model look very solid. Additionally,'
                ' the average RMSE on the test set was equal to 586.4, which is, considering the large '
                'number of cases, not too bad. It is now interesting to dive into trained the trained parameters'
                ' of the CAR model. For each state $i$, a local regression coefficient $\\beta_i$'
                ' and a spatial parameter $z_i$. The following plot illustrates how these parameters are spread over the'
                'states. Note that the marker size in the scatter plot is proportional to the total number of cases that were last '
                'recorded in that state.')

    img = Image.open(config.notebooks/'streamlit-app'/'images'/'StatesAnnotCar1.png')
    st.image(img, caption='', format='PNG', width=650)

    st.markdown('From the figure, we would hope to discover that states with a similar spread of Covid-19, would'
                ' be grouped together. This is not entirely the case, but it is cool to see how for instance the '
                'New York is clearly an outlier, which makes total sense given its extreme Covid history.')

    st.markdown('Now the model has been tested on a relatively easy task, we can now try to predict the number of '
                'Covid cases per state 5 days ahead in time. For this we will also increase the amount of lags included '
                'as local regression predictors to 5. The resulting preditions for 10 example states are illustrated below: ')

    img = Image.open(config.notebooks/'streamlit-app'/'images'/'CARLLAG5.png')
    st.image(img, caption='CAR predictions for 5 days ahead', format='PNG', width=700)

    state = st.selectbox('Select state for which you wish to see the 5 days ahead CAR predictions. It may take a second.',
		 covid_data.columns)

    CAR_model_lag5 = pd.read_csv(config.data / 'processed' / 'CARModelLag5_V2')
    CAR_model_lag5.set_index('Unnamed: 0', inplace = True)
    params_dict = dict()
    for k, state_name in enumerate(covid_data.columns):
        params_k = [CAR_model_lag5['mean']['beta0[{}]'.format(k)],
                    CAR_model_lag5['mean']['phi[{}]'.format(k)]]
        params_dict[state_name] = params_k

    def lag5_pred(state):
        # predict
        pred_state = []
        for i, day in enumerate(range(start + lag, len(covid_data) - pred_ahead)):
            params = params_dict[state]
            lag_X = [covid_data[state][day - step]/covid_data[state][day - step -1] for step in range(1, lag+1)]
            log_pred = params[0]
            for j in range(lag):
                log_pred += CAR_model_lag5['mean']['betas[{}]'.format(j)]*lag_X[j]
            log_pred += params[1]
            pred = np.exp(log_pred)
            T_pred = pred*covid_data[state][day]
            pred_state.append(T_pred)

        rmse = np.sqrt(mean_squared_error(pred_state,
                                 covid_data[state].iloc[start + lag: len(covid_data) - pred_ahead]))
        rmse = np.round(rmse, 3)

        return pred_state, rmse

    # plot
    pred_ahead = 5
    lag = 5
    start = 57
    pred_state, rmse = lag5_pred(state)
    fig2 = plt.figure(figsize = (12,8))
    plt.plot(range(start, len(covid_data[state])), covid_data[state][start:], label = 'Real data')
    plt.plot(range(start+lag+pred_ahead, len(covid_data[state])),
             pred_state, label = '5 days ahead CAR prediction - RMSE = {}'.format(rmse))
    plt.title('CAR predictions of Covid cases for {}'.format(state), fontsize = 20)
    plt.xlabel('Day since Jan 21 2020', fontsize = 20)
    plt.ylabel('# cases', fontsize = 20)
    plt.legend(fontsize = 20)
    st.pyplot(fig2)

    st.markdown('Clearly, the results are not as good as with the one day prediction, but it still'
                'seems pretty decent. The RMSE on the test set in this case is equal to 11488.5. '
                ''
                'Note that prediction is only one of the advantages of the CAR model. By having learned all '
                'the spatial parameters, we can investigate the spatial dependence of Covid predictions. '
                'For instane, the following figure illustrates how the predictions in the state New York are'
                'consistently underpredicting because its neighboring states have significantly lower cases.')

    img = Image.open(config.notebooks/'streamlit-app'/'images'/'CAR_NY_Neighb.png')
    st.image(img, caption='CAR prediction for New York and its neighbors', format='PNG', width=700)

    st.markdown('While this interdependency of states might seem as a disadvantage at first, it could '
                'potentially help to identify whether states have been underreporting the number of Covid cases '
                'or not. In order to test this, we can have a look at the positive test rates for each state. '
                'The dataset below contains the current positive test rate for Covid for all states. '
                'Source: https://covidtracking.com/api. ' )

    test_data = pd.read_csv(config.data / 'processed' / 'TestRates.csv')
    st.write(test_data)

    state_name2abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS',
    'Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
    'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM',
    'New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
    'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT',
    'Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'}

    abbrev2state_name = {state_name2abbrev[key]: key for key in state_name2abbrev.keys()}
    test_rates = dict()

    for i, state_abbrev in enumerate(test_data['state']):
        test_rate = 100*test_data['positive'].iloc[i]/test_data['total'].iloc[i]
        if state_abbrev in abbrev2state_name.keys():
            state_name = abbrev2state_name[state_abbrev]
            test_rates[state_name] = test_rate
        #else:
            #print(state_abbrev)

    fig5 = plt.figure(figsize = (20,15))
    plt.bar(range(len(test_rates)), test_rates.values())
    plt.xticks(range(len(test_rates)), test_rates.keys(), rotation = 'vertical', fontsize = 10)
    plt.xlabel('State', fontsize = 20)
    plt.ylabel('Positive test rate [%]',  fontsize = 20)
    plt.title('Positive test rate for Covid for all US states',  fontsize = 20)
    st.pyplot(fig5)

    st.markdown('The positive test rate can clearly differ significantly from state to state.'
                'When its value is high for a specific state, the probability that the number of '
                'actual cases of Covid-19 in that state is higher than the reported value, is higher.'
                'We therefore came up with a measure of overpredicting of our CAR model, being the '
                'average difference of the prediction and the ground truth. When this is positive, '
                'our CAR model is actually suggesting that, based on its spatial knowledge, the state '
                'is more likely to have more cases in reality than is currently reported. Therefore, '
                'it would make sense that our overpredicting rate is positively correlated with the positive '
                'test rate. The following figure illustrates that this is slightly the case.')

    img = Image.open(config.notebooks/'streamlit-app'/'images'/'CARTestOverpred.png')
    st.image(img, format='PNG', width=700)

    st.markdown('In conclusion, we have investigated the performance of a classical Bayesian model,'
                'the Conditional Autoregressive model, to the spatial and temporal spread of Covid-19.'
                'Using simple LASSO regression on the large dataset on the flu, we were able to come up with'
                'a reasonable set of neighbors for each state in the context of deseases. We then build the CAR '
                'model using this weight matrix W to predict the total number of Covid cases for'
                ' both 1 and 5 day(s) ahaead for each US state. While the results were not bad,'
                'we have to admit that the prediction performance was not great either. Given the monotonicity of the data,'
                'we could easily achieve the same performance with simple autoregression. We did find additional value '
                'in using the CAR model when we wish to understand the spatial interdependency of the spread of Covid. '
                'From the last graph, it was for instance clear that our model has the potential to estimate whether '
                'a state is under-reporting the total number of cases or not. For us, this was a reasonable way'
                ' of embedding information on the flu in a Bayesian model for Covid, that can lead to '
                'interpretable results.')


if __name__ == "__main__":
    main()




