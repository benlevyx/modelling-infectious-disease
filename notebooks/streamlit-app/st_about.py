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
from covid_flu import config


def crop_center(pil_img):
    img_width, img_height = pil_img.size
    img_length = min(img_width, img_height)
    return pil_img.crop(((img_width - img_length) // 2,
                         (img_height - img_length) // 2,
                         (img_width + img_length) // 2,
                         (img_height + img_length) // 2))


def main():
    with open(str(config.streamlit / 'about.css')) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    st.title("About us")

    st.write("""
    We are a group of Data Science Master's students at the Institute for Applied Computational
    Science (IACS) at Harvard University.""")

    st.write("### The team")

    imfiles = ["ben", "dimitris", "matthieu", "will"]
    images = [crop_center(Image.open(config.images / 'about' / f'{f}.jpg')) for f in imfiles]
    captions = [
        "Benjamin Levy (https://benlevyx.github.io)",
        "Dimitris Vamvourellis (https://github.com/dvamvourellis)",
        "Matthieu Meeus (https://github.com/matthieumeeus)",
        "Will Fried (https://github.com/williamfried)"
    ]
    st.image(images, width=300, caption=captions)


    # st.image(Image.open(config.images / 'about' / 'ben.jpg'), width=300, caption="Benjamin Levy(https://benlevyx.github.io)")
    # st.image(Image.open(config.images / 'about' / 'dimitris.jpg'), width=300, caption="Dimitris Vamvourellis")
    # st.image(Image.open(config.images / 'about' / 'will.jpg'), width=300, caption="Will Fried")
    # st.image(Image.open(config.images / 'about' / 'matthieu.jpg'), width=300, caption="Matthieu Meeus")