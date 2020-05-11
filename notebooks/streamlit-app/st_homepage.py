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
    st.title('Homepage')

    st.write('#### Authors: Dimitris Vamvourellis, Benjamin Levy, Will Fried, Matthieu Meeus')

    st.write("""
    ## Goals and problem statement
    
    The overarching goal of this project is to build models to predict seasonal
    influenza and COVID-19, hopefully using information about the former to augment
    our predictions and inferences about the latter. The project consists of the
    following components:
    
    **1. Exploratory Data Analysis**
    
    We examine the time series data for seasonal flu, noting interesting spatial
    and temporal patterns.
    
    **2. Data preprocessing**
    
    We transfer the data to a usable form, filling in unwanted gaps in the
    flu time series using a gaussian process.
    
    **3. Flu inference**
    
    We build a Bayesian model that aims to estimate the spatial dependencies
    between states and explain those dependencies in terms of underlying features
    of those states.
    
    **4. Flu forecasting**
    
    We use sequence-to-sequence (seq2seq) recurrent neural networks to forecast
    the seasonal flu,
    
    **5. COVID-19 forecasting**
    
    We construct a spatial Bayesian model for COVID-19, using geographic
    patterns to inform the spatial dependencies.
    
    **6. COVID-19 transfer learning**
    
    We pre-train seq2seq models on the seasonal flu and examine whether this
    pre-training can be used to improve COVID-19 forecasting using transfer
    learning (TL).
    """)

    st.write("""
    **The github repo for this project can be found [here](https://github.com/benlevyx/modelling-infectious-disease/tree/master).**
    
    The notebooks for each section are linked in the respective writeups.
    
    To navigate to different sections, use the dropdown box on the left! :point_left:
    """)


