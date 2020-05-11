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
    with (config.streamlit / 'about.css').open('r') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    st.title("About us")

    st.subheader("""We are a team of Data Science Master's students in the Institute for Applied Computational Science (IACS) at Harvard University.""")

    st.markdown(f"""
    <div class="container">
      <div class="box">
        <img src='{str(config.images / 'about' /  'dimitris.jpg')}'>
      </div>
      <div class="box">
        <img>
      </div>
      <div class="box">C</div>
      <div class="box">D</div>
    </div>
    """,unsafe_allow_html=True)