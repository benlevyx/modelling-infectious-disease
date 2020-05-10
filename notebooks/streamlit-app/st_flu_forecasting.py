import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
from covid_flu import config, utils, time_series, plotting, models


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import csv

plt.rcParams.update({'font.size': 18})
import plotly.express as px  # Be sure to import express
from covid_flu import config

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
        
         As explained above, the Seq2Seq model will not output a vector sequence directly 
        as the naive vector-output LSTM. Instead, by using an LSTM model in the decoder, we allow the model to both know 
        what was predicted for the prior week in the sequence and accumulate internal state while outputting the predictions for 
        the next T weeks. In this way, the model makes multi-steps predictions in a more structured way. At the same time, 
        our experiments showed that a basic Attention mechanism did improve the performance. 
        
        The code used to develop the more complex Seq2Seq architectures can be accessed [here](https://github.com/benlevyx/modelling-infectious-disease/blob/master/covid_flu/covid_flu/models.py).
        The notebook with the results for each model and the complete analysis can be accessed [here](https://github.com/benlevyx/modelling-infectious-disease/blob/master/notebooks/flu_forecasting_final.ipynb). 
        """
             )

    img = Image.open(config.notebooks/'streamlit-app'/'images'/'seq2seq.png')
    st.image(img, caption='test', format='PNG', width=1000)



if __name__ == "__main__":
    main()


