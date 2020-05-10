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
    Conditional autoregressive model | Sequence-to-sequence models
    
    
    """)