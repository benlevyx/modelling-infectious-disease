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

def main():
    st.title('Conclusion')

    st.write("""
    #### We return to the question: can we use the flu to predict COVID-19?
    
    In our EDA, we noticed two main patterns in the flu data: **spatial** and
    **temporal**. As such, we attempted to exploit these patterns to see
    if COVID-19 might exhibit a similar-enough trend that a predictive model
    based on the flu data could achieve better performance than a mdoel
    based solely on COVID data alone.
    
    Spatial | Temporal
    ------- | --------
    Bayesian hierarchical model | Recurrent neural networks
    Conditional autoregressive model (CAR) | Sequence-to-sequence models
    
    In our analyses, we found that these kinds of models were quite good
    at predicting the seasonal flu. We achieved very low prediction error
    (out of sample) for predicting seasonal flu using a seq2seq model that
    took advantage of the clear cyclical trends over the ten years of flu data
    we had.
    
    However, when it came to predicting COVID-19 by exploiting the spatiotemporal
    patterns in the seasonal flu, we achieved mixed results. The CAR model did
    not clearly outperform a simple AR(1) (autoregressive) model, which was our
    baseline. As well, the seq2seq models, pre-trained on flu data, did not
    result in plausible mid-to-long-term forecasts.
    
    Yet these COVID-19 models nonetheless revealed interesting findings that merit
    further exploration. For the CAR model, we found that the model consistently
    over-estimated the number of cases in states with higher rates of positive
    tests. We can tentatively interpret this to mean that, based on the spatial
    patterns from the flu, we would expect that states with poorer test coverage
    (i.e. they are not testing widely enough to see lots of negative results)
    actually have far more cases of COVID-19 than reported.
    
    As well, the exercise of pre-training a seq2seq model on flu and then transferring
    it to COVID-19 did demonstrate the power of this technique, as the best RNN
    model we tested for COVID-19 was the one that was trained on the flu and then
    fine-tuned on COVID-19.
    
    ## Next steps
    
    Much of the work done in this project has been preliminary and merits further
    exploration. Here are some areas that we hope to pursue in the future:
    
    * Tuning the parameters of both the hierarchical Bayesian model and the CAR
    model to ensure convergence
    * Exploring different RNN architectures that can better take into account
    domain knowledge about COVID-19, such as the parameter estimates from the
    literature
    * Trying other pre-trained models or external datasets that might be better
    suited for transfer learning with COVID-19
    
    Thank you for your time! We hope you enjoyed reading these results as much
    as we enjoyed conducting this project! Stay safe and stay healthy!
    """)