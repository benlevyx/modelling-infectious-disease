import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import csv

plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config, utils, models, plotting, time_series


# Some hyperparameters
HISTORY_SIZE = 25
TARGET_SIZE = 5
BATCH_SIZE = 64
TEST_SIZE = 0.3
EPOCHS = 20
NUM_ENCODER_LAYERS = 2
HIDDEN_SIZE = 16
PRE_OUTPUT_DENSE_SIZE = 16

scaler_flu = StandardScaler()
scaler_covid = StandardScaler()


@st.cache
def load_data():
    flu_data = utils.load_flu_data()
    covid_data = utils.load_covid_data()
    return flu_data, covid_data


class Container:
    pass


def load_globals():
    D = Container()
    df_flu, df_covid = load_data()
    df_covid = df_covid[df_covid['total_cases'] >= 10]
    states = df_covid['state'].unique().tolist()

    df_flu['wili_sc'] = utils.scale_data(df_flu['wili'].values, scaler_flu)
    df_covid['cases_sc'] = utils.scale_data(df_covid['cases'].values, scaler_covid)

    D.states = states
    D.df_flu = df_flu
    D.df_covid = df_covid

    # Preparing tensorflow datasets
    # Making train/val/test split
    ds_train_flu, ds_val_flu, ds_test_flu, data_dict_flu = time_series.prepare_data(df_flu['wili_sc'].values,
                                                                                    df_flu['state'].values,
                                                                                    history_size=HISTORY_SIZE,
                                                                                    target_size=TARGET_SIZE,
                                                                                    batch_size=BATCH_SIZE,
                                                                                    test_size=TEST_SIZE,
                                                                                    teacher_forcing=True)

    D.data_dict_flu = data_dict_flu

    ds_train_covid, ds_val_covid, ds_test_covid, data_dict_covid = time_series.prepare_data(df_covid['cases_sc'].values,
                                                                                            df_covid['state'].values,
                                                                                            history_size=HISTORY_SIZE,
                                                                                            target_size=TARGET_SIZE,
                                                                                            batch_size=BATCH_SIZE,
                                                                                            test_size=TEST_SIZE,
                                                                                            teacher_forcing=True)

    D.data_dict_covid = data_dict_covid
    return D

def load_model(name):
    model = models.Seq2Seq(history_length=HISTORY_SIZE,
                           target_length=TARGET_SIZE,
                           hidden_size=HIDDEN_SIZE,
                           pre_output_dense_size=PRE_OUTPUT_DENSE_SIZE,
                           num_encoder_layers=NUM_ENCODER_LAYERS)
    utils.load_weights(model, name)
    return model


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


def data(D):
    st.header("Data loading and preparation")

    st.write("**Flu data**")
    st.write(D.df_flu)
    st.write("**COVID-19 data:** we preprocess the COVID-19 data so that, like the flu"
             "data, it represents new cases (lags) rather than total cases. This is"
             "the initial step we can take to make the two datasets more similar.")
    st.write(D.df_covid)

    st.write("Graphing the two:")

    fig = plt.figure(figsize=(16, 6))
    plt.title("Weekly cases of influenza")
    for state in D.df_flu['state'].unique():
        subset = D.df_flu.query('state == @state')
        plt.plot(subset['time'], subset['wili'])
    plt.ylabel("Confirmed cases")
    plt.xlabel("Year-Week of year")

    dates = np.unique(D.df_flu['time'])
    date_idxs = np.arange(0, len(dates), 50)
    plt.xticks(ticks=date_idxs, labels=dates[date_idxs])
    st.pyplot(fig)

    fig = plt.figure(figsize=(16, 6))
    plt.title("New confirmed cases of COVID-19")
    for state in D.df_covid['state'].unique():
        subset = D.df_covid.query('state == @state').query('date > @pd.to_datetime("2020-03-15")')
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


def flu_model(D):
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

    st.write("After training the model for 20 epochs, we can look at some "
             "predictions for 5 weeks ahead:")

    model = load_model('tl__flu')

    state_chosen = st.selectbox("Choose a state to see predictions for (may take a minute to recompute)", D.states, key='flu')

    fig, axs = plt.subplots(3, 1, figsize=(16, 24))
    for i, split in enumerate(['train', 'val', 'test']):
        plotting.plot_seq2seq_preds(model, D.data_dict_flu, state_chosen,
                                    TARGET_SIZE, scaler=scaler_flu,
                                    split=split, ax=axs[i])
        axs[i].set_title(f'{split} data')
        if i > 0:
            axs[i].get_legend().remove()

    st.pyplot(fig)


def covid_models(D):
    st.header("Transfer learning for forecasting COVID-19")
    st.write("""
    We will try to model COVID in four ways:

1. Identical model to flu, training from scratch
2. The flu model without any retraining
3. The flu model with the FCN head fine-tuned
4. The flu model, trained on the flu data resampled to daily frequency, with the FCN head fine-tuned on COVID
""")
    st.write("Here are the resulting predictions for these models:")

    model_names = ['scratch', 'fine-tuned', 'all-tuned']
    model_fnames = ['tl__covid_scratch', 'tl__covid_finetune', 'tl__covid_tuneall']
    models = [load_model(m) for m in model_fnames]

    state_chosen = st.selectbox("Choose a state to see predictions for (may take a minute to recompute)", D.states, key='covid')

    fig, axs = plt.subplots(3, 1, figsize=(16, 24))
    for i, split in enumerate(['train', 'val', 'test']):
        X, tpre, y, tpost = plotting.plot_seq2seq_preds(models[i], D.data_dict_covid, state_chosen,
                                                        TARGET_SIZE, scaler=scaler_covid,
                                                        split=split, ax=axs[i], return_truth_only=True)

        axs[i].plot(tpre, X, color='k', label='History')
        axs[i].plot(tpost, y, color='blue', label='Ground truth')

        for model_name, model in zip(model_names, models):
            yhat, tpost = plotting.plot_seq2seq_preds(model, D.data_dict_covid, state_chosen,
                                                      TARGET_SIZE, scaler=scaler_covid,
                                                      split=split, ax=axs[i], return_preds_only=True)
            axs[i].plot(tpost, yhat, 'x', label=model_name)

        axs[i].set_title(f'{split} data')
        axs[i].legend()

    st.pyplot(fig)


def conclusion():
    rmses = [963.9749629416106,
             1218.7620641410642,
             711.040621587595,
             1218.7620641410642,
             962.7328324085669]
    names = ['Base flu model', 'Base COVID model',
             'Fine-tuned', 'Full tuned', 'Resampled']

    df_rmse = pd.DataFrame({
        'Model': names,
        'RMSE': rmses
    }).set_index('Model')

    st.header("Conclusion")
    st.write("We can compare the different models using root mean squared error (RMSE):")

    st.table(df_rmse)

    st.write("while no single model yielded particularly impressive results "
             "that rival any of the state of the art epidemiological models, "
             "using the flu time series to pre-train the seq2seq model does "
             "in fact yield a performance gain over just training on COVID "
             "alone! Furthermore, freezing the base and just fine-tuning the "
             "head is significantly better than full retraining, supporting "
             "our hypothesis that reducing the number of parameters to train "
             "helps the model a lot!")

    st.write("For the full analysis and code, as well some exploration of "
             "using state embeddings to build a conditional seq2seq model, "
             "please check out the notebook used to build these models: "
             "[COVID-19 Transfer Learning](https://github.com/benlevyx/modelling-infectious-disease/blob/master/notebooks/covid-transfer-learning-final.ipynb)")


def main():
    introduction()
    D = load_globals()
    data(D)
    flu_model(D)
    covid_models(D)
    conclusion()
