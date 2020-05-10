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

