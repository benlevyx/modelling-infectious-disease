from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


def display_all_rows(df):
    with pd.option_context("display.max_rows", None):
        display(df)


def display_all_cols(df):
    with pd.option_context("display.max_columns", None):
        display(df)


def display_all(df):
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(df)


def load_flu_data():
    return pd.read_csv(config.processed / 'flu_ground_truth_imputed.csv')


def load_covid_data():
    covid_data = pd.read_csv(config.raw / 'Covid_data.csv', index_col=0)
    covid_data = covid_data.melt(id_vars=['date'])
    covid_data.columns = ['date', 'state', 'cases']
    covid_data = covid_data.fillna(0.)
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    return covid_data


def scale_data(x: np.ndarray, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    x_sc = scaler.fit_transform(x.reshape(-1, 1)).flatten()
    return x_sc
