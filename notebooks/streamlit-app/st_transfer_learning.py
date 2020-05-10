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

