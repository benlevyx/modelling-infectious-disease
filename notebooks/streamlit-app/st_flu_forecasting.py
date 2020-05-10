import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
from covid_flu import config, utils, time_series, plotting, models


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import csv

plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config

def load_keras_model(model_name):
    # Load json and create model
    json_file = open(config.models / (model_name+'.json', 'r'))
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights(config.models/ (model_name + '.h5'))
    return model


def predict(encoder_model, decoder_model, target_size, X):

    pred_steps = target_size
    if X.ndim == 2:
        X = np.expand_dims(X, 0)

    states_value = encoder_model.predict(X)
    target_seq = np.zeros((X.shape[0], 1, 1))
    target_seq[:, 0, 0] = X[:, -1, 0]
    decoded = np.zeros((X.shape[0], pred_steps, 1))

    for i in range(pred_steps):
        output, h, c = decoder_model.predict([target_seq] + states_value)
        decoded[:, i, 0] = output[:, 0, 0]
        target_seq = np.zeros((X.shape[0], 1, 1))
        target_seq[:, 0, 0] = output[:, 0, 0]

        states_value = [h, c]

    return decoded


def plot_multiple_steps_preds_for_state(X_test, y_test, states_test, encoder_model, decoder_model, target_size, state, train_size, scaler=None):
    # plot the preds for target_size weeks ahead skipping every target_size inputs
    preds = predict(encoder_model, decoder_model, target_size, X_test[states_test == state])
    y = y_test[states_test == state]
    if scaler != None:
        y = scaler.inverse_transform(y_test[states_test == state])
        preds = scaler.inverse_transform(preds)

    multiple_step_preds = preds[::target_size, :].flatten()
    y = y[::target_size, :].flatten()

    idx = int(y.shape[0]*train_size)
    x = np.arange(y.shape[0])

    plt.figure(figsize=(10, 8))
    plt.plot(x,y, 'b-', label='True')
    plt.plot(x[:idx], multiple_step_preds[:idx], 'g-', label='Predicted Train')
    plt.plot(x[idx:], multiple_step_preds[idx:], 'r-', label='Predicted Test')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('WILI')
    plt.title(f'True vs {target_size}-steps prediction for {state}')
    st.pyplot()


@st.cache
def load_data():
    df_flu = pd.read_csv(config.processed / 'flu_ground_truth_imputed.csv')
    X = df_flu['wili'].values
    groups = df_flu['state'].values
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X.reshape(-1, 1)).flatten()

    HISTORY_SIZE = 50
    TARGET_SIZE = 5

    ds_train, ds_val, ds_test, data_dict = time_series.prepare_data(X_scaled,
                                                                    groups,
                                                                    history_size=HISTORY_SIZE,
                                                                    target_size=TARGET_SIZE,
                                                                    batch_size=64,
                                                                    test_size=0.2,
                                                                    teacher_forcing=True
                                                                    )

    X_all = np.concatenate((data_dict['X_train'], data_dict['X_val'], data_dict['X_test']))
    y_all = np.concatenate((data_dict['y_train'], data_dict['y_val'], data_dict['y_test']))
    states_all = np.concatenate((data_dict['states_train'], data_dict['states_val'], data_dict['states_test']))
    return X_all, y_all, states_all, sc


def main():
    st.title('Forecasting Flu with Recurrent Neural Networks')
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    st.write("""
        ## Summary
        The goal of this exploration is to build recurrent neural networks (RNN) for forecasting flu rates across US states. 
        Ultimately, our goal is to build an accurate model for multi-step predictions. For instance, we argue that 
        knowing the projected trends in flu rates in advance for the next 5 or 10 weeks is significantly more important than
        predicting flu rate for the next week with the highest accuracy. Such a tool would be invaluable for public health officials
        in order to know in advance if there are projected spikes in flu activity over the next 1-2 months in order to plan ahead 
        accordingly and take preventive measures.
        
        As shown in the EDA section, flu rates exhibits notable seasonal patterns across states over the last 10 years. In other
        words, the history seems to be repeated. LSTM networks are a natural choice for this problem setting, since they learn
        how to map long sequences of historical observations to future predictions. For the purpose of this analysis, we only used
        the history of WILI rates across states as a feature. However, LSTMs can also handle multi-dimensional features.
        In future steps, we will potentially incorporate other correlated predictors along with historical WILI observations.
        
        In our analysis, we explored a variety of models ranging from simple LSTM models to more complex RNN structures. Specifically we explored:

        - Simple LSTM for one step predictions.
        - Simple LSTM with vector output for multi-step predictions
        - Seq2Seq encoder-decoder architectures for multi-step predictions
        - Seq2Seq with attention for multi-step predictions
        
        Based on our analysis, the Seq2Seq RNNs achieved the highest performance in predicting the flu rates for the next 5 or 10 weeks. 
        Thus, we discuss this modelling approach in more detail here. The complete analysis along with the results for each of the above models 
        can be accessed [here](https://github.com/benlevyx/modelling-infectious-disease/blob/master/notebooks/flu_forecasting_final.ipynb). 
        The code used to develop the more complex Seq2Seq architectures can be accessed [here](https://github.com/benlevyx/modelling-infectious-disease/blob/master/covid_flu/covid_flu/models.py).
        
        
        ## Seq2Seq architecture
        With Sequence-to-sequence (Seq2Seq) architectures, we can train models to convert sequences from one domain to sequences in another domain. For instance, 
        one of the main application fields for Seq2Seq models is machine translation, where sentences (sequences of words) in one language are converted to sentences
        in another language. The same idea can be applied for time seried predictions, to convert the history of observations of a variable (sequence of H steps) to a sequence 
        of predictions in the future (sequence of T steps). For example, in this problem setting, we are using Seq2Seq architectures to map a sequence of 50 historical weekly 
        flu rate observations to a sequence of 5 or 10-week future predictions.
        
        A Seq2Seq model is composed an encoder and a decoder. The encoder, usually composed of stacked LSTM layers, processes the input sequence 
        and returns its internal state, while the outputs of the encoder RNN are discarded. This state serves as the "context", encapsulating all 
        the information from the input sequence in a compact way. The decoder is another RNN layer (or a stack of more RNN layers) which is trained 
        to predict the next T elements of the sequence, given the context from the encoder. This means that the model will not output a vector sequence directly as 
        the naive vector-output LSTM. Instead, by using an LSTM model in the decoder, we allow the model to both know what was predicted for the prior week in the sequence 
        and accumulate internal state while outputting the predictions for the next T weeks.
        """
        )

    img = Image.open(config.notebooks/'streamlit-app'/'images'/'seq2seq.png')
    st.image(img, caption='Seq2Seq architecture (source: https://github.com/Arturus/kaggle-web-traffic/blob/master/images/encoder-decoder.png)', format='PNG', width=800)

    st.write("""
        As explained above, the Seq2Seq model will not output a vector sequence directly 
        as the naive vector-output LSTM. Instead, by using an LSTM model in the decoder, we allow the model to both know 
        what was predicted for the prior week in the sequence and accumulate internal state while outputting the predictions for 
        the next T weeks. In this way, the model makes multi-steps predictions in a more structured way taking both a longer history of observations
        and the most recent prediction into account.
    """
    )
    HISTORY_SIZE = 50
    TARGET_SIZE = 5
    seq2seq_model = models.Seq2Seq(history_length=HISTORY_SIZE,
                                   target_length=TARGET_SIZE,
                                   hidden_size=32,
                                   num_encoder_layers=3,
                                   num_decoder_layers=1,
                                   pre_output_dense_size=16
                                   )

    encoder_model = seq2seq_model.encoder_model
    encoder_model.load_weights(str(config.models / 'seq2seq_50_5_encoder.h5'))

    st.write('Encoder Model Summary')
    #st.write(encoder_model)

    decoder_model = seq2seq_model.decoder_model
    decoder_model.load_weights(str(config.models / 'seq2seq_50_5_decoder.h5'))

    st.write('Decoder Model Summary')
    #st.write(decoder_model)


    X_all, y_all, states_all, sc = load_data()

    state_chosen = st.selectbox('Select state that you want to see forecasts for.', np.unique(states_all))

    plot_multiple_steps_preds_for_state(X_all, y_all, states_all, encoder_model, decoder_model, 5, state_chosen, 0.8, sc)






if __name__ == "__main__":
    main()


