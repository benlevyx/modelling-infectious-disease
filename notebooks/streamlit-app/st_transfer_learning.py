import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import csv

plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config, utils


def introduction():
    st.title('Transfer learning for forecasting COVID-19')
    st.write("""
        In this section, we examine another strategy of using flu data to help us
        to model COVID-19: **Transfer learning**
        """)
    st.write("""
        As already seen in our notebook exploring how RNNs can be used to predict flu (found [here](#)), these kinds of algorithms can be quite effective for even long-term predictions. However, a major limitation of sequence-to-sequence (seq2seq) models, and many powerful ML algorithms more broadly, is that they require a large amount of data. This is because the number of parameters (weights) can run into the tens of thousands even for simple models, let alone massive models with hundreds of millions of parameters.

    One way to deal with the problem of data scarcity is transfer learning (TL). This is a deceptively simple technique whereby a model $M$ that has been trained using dataset $D$ for task $T$ is then re-used, entirely or in part, with fine-tuning or as-is, for another dataset $D'$ and/or task $T'$. For instance, in AC209b, we used a pre-trained MobileNet CNN to perform image classification.

    There are several ways to do TL and which one we use depends heavily on two factors:

    * How different our dataset $D'$ is from the dataset the model was trained on $D$
    * How different our task $T'$ is from the task the model was trained on $T'$

    If the tasks $T$ and $T'$ are very different, then it is likely necessary to modify the model architecture. In the case of a CNN trained for image classification, it is common practice to freeze the convolutional part, which can be seen as a feature extractor, and to change the fully connected network (FCN), which can be seen as a classifier. For some large pre-trained models, such as ImageNet or MobileNet, the convolutional base can be sufficiently complex to extract a rich enough set of features for a new FCN head to work with. This is referred to as *representation learning* -- using a neural network to convert the input into an abstract representation that is more useful for downstream tasks.

    If datasets $D$ and $D'$ are very different, but $T$ and $T'$ are similar, then it is likely sufficient to fine-tune part or all of the model, while maintaining the same architecture.

    In general, the more similar $D$ and $D'$, as well as $T$ and $T'$, then the more successful TL is likely to be.

    #### Problem formulation

    We have the following two datasets:

    * $D$ = Reported weekly seasonal influenza cases from the last 10 years in the United States
    * $D'$ = Reported daily COVID-19 cases from the beginning of 2020 in the United States

    We have only one task: univariate time series forecasting.

    We will attempt to apply TL by using a model trained on the flu data and then transferring it to COVID-19 data.

    #### Outline of analysis

    1. Data loading and preparation
    2. Training seq2seq model on influenza
    3. Transferring seq2seq model to COVID-19
        1. No TL
        2. Fine-tuning just the FCN head
        3. Fine-tuning the entire model
        """)


def data():
    st.header("Data loading and preparation")

    @st.cache
    def load_data():
        flu_data = utils.load_flu_data()
        covid_data = utils.load_covid_data()
        return flu_data, covid_data

    df_flu, df_covid = load_data()

    st.write("**Flu data**")
    st.write(df_flu)
    st.write("**COVID-19 data:** we preprocess the COVID-19 data so that, like the flu"
             "data, it represents new cases (lags) rather than total cases. This is"
             "the initial step we can take to make the two datasets more similar.")
    st.write(df_covid)

    st.write("Graphing the two:")

    fig = plt.figure(figsize=(16, 6))
    plt.title("Weekly cases of influenza")
    for state in df_flu['state'].unique():
        subset = df_flu.query('state == @state')
        plt.plot(subset['time'], subset['wili'])
    plt.ylabel("Confirmed cases")
    plt.xlabel("Year-Week of year")

    dates = np.unique(df_flu['time'])
    date_idxs = np.arange(0, len(dates), 50)
    plt.xticks(ticks=date_idxs, labels=dates[date_idxs])
    st.pyplot(fig)

    df_covid = df_covid[df_covid['total_cases'] >= 10]
    fig = plt.figure(figsize=(16, 6))
    plt.title("New confirmed cases of COVID-19")
    for state in df_covid['state'].unique():
        subset = df_covid.query('state == @state').query('date > @pd.to_datetime("2020-03-15")')
        if (subset['cases'] > 4000).any():
            plt.plot(subset['date'], subset['cases'], label=state)
        else:
            plt.plot(subset['date'], subset['cases'])
    plt.ylabel("Confirmed cases")
    plt.xlabel("Date")
    plt.legend()
    st.pyplot(fig)

    st.write("""
    #### Notable differences between the two datasets

The flu and COVID-19 datasets might be alike in that they are both diseases that are quite contagious and can follow exponential growth trends, it is evidence from the two time series that they are both quite different in terms of their progression in the population. There are also important differences between the datasets themselves, irrespective of the underlying diseases.

* **Temporal resolution**: Flu data are collected weekly, allowing for more sharp differences between individual time points, whereas COVID data are reported daily
* **Trajectory**: The flu tends to follow a pattern of steep peaks spaced out by wide troughs, which repeats throughout the year. COVID, on the other hand, exhibits a slow and steady
* **Completeness**: The flu data cover 10 years completely, whereas the COVID data do not even cover half a year. Thus, the patterns that might be noticed in the flu data are not available for forecasting COVID.

These differences will need to be mitigated where possible or accepted as limitations in this methodology. Later on, we will attempt to mitigate the effect of the first (temporal resolution) whereas the other two are likely better dealt with through other methods (such as agent-based models).
    """)


def flu_model():
    st.header("Building a model for flu forecasting")
    st.write("""
    Building on the work on forecasting flu using RNNs, 
    we use a seq2seq RNN consisting of a 3-layer LSTM encoder and a 1-layer 
    decoder. We train using teacher forcing, where the decoder predicts a 
    single point at a time and the next point fed to the decoder is the true 
    value of the time series, rather than what was predicted on the previous step.
    """)
    img = Image.open(config.images / 'transfer-learning' / 'seq2seq_flu.png')
    st.image(img, caption='The basic seq2seq model for flu', format='PNG', width=600)

    st.write("After training the model for 20 epochs, we can look at some"
             "predictions  for 5 weeks ahead:")

    st.write("**INSERT GRAPH OF FLU PREDICTIONS HERE**")


def covid_models():
    st.header("Transfer learning for forecasting COVID-19")
    st.write("""
    We will try to model COVID in four ways:

1. Identical model to flu, training from scratch
2. The flu model without any retraining
3. The flu model with the FCN head fine-tuned
4. The flu model, trained on the flu data resampled to daily frequency, with the FCN head fine-tuned on COVID
""")
    st.write("Here are the resulting predictions for those same 5 states:")
    print("**INSERT INTERACTIVE GRAPH OF COVID PREDICTIONS HERE**")


def conclusion():
    pass


def main():
    introduction()
    data()
    flu_model()
    covid_models()
