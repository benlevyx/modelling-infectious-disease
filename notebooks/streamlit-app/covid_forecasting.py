import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from sklearn.linear_model import Lasso

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import csv

plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config

def main():
    st.title('Covid Forecasting')
    st.write('In this section, we will explore how we can gain knowledge from 10 years of flu data to '
             'better predict the spread of Covid-19 over the US states. Two main approaches will be attempted: '
             'one Conditional Autoregressive Model and one transfer learning RNN approach (BEN)')
    st.subheader('1. Get the data')
    st.write('First, we should have a look at the data. For the Covid-19 data, we found that the New York Times '
             'released a publicly available dataset (source:https://data.humdata.org/dataset/nyt-covid-19-data). It contains the daily number of recorded cases per US '
             'state since January 21 2020, which was the date of the first confirmed case in the US.')\

    @st.cache
    def fetch_and_clean_data():
        flu_dir = config.data / 'raw'
        filename = 'Covid_data.csv'

        df = pd.read_csv(flu_dir / filename)
        return df

    covid_data = fetch_and_clean_data()

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
             '$\epsilon_i \sim N(0, \sigma_{\epsilon_i}^2)$ \n The spatial factors can then still be determined'
             ' in the same way: \n'
             '$$z_i | z_j , i \\neq j \sim N(\sum_{j} c_{ij} z_j, m_{ii})$$'
             'Following standard practice, we can define the matrix $C = \\rho W$. Now it comes down to '
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

    st.markdown('This seems to make sense! So let\s continue using these coefficients.')

    covid_data.drop(['Unnamed: 0', 'date', 'Alaska','District of Columbia', 'Hawaii'], axis = 1,
                    inplace = True)

    st.markdown('Using this, we can now start building the CAR model. We will first do so by predicting '
                'one day ahead using one day lag. The model can then be implemented with an MCMC using PyMC3. '
                'For details on the implementation, it is recommended to have a look at the notebook on Github.')


    
if __name__ == "__main__":
    main()




