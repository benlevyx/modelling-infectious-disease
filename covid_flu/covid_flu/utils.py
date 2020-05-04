import re

from IPython.display import display
import pandas as pd

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
    flu_dfs = []
    for fpath in (config.raw / 'flu_ground_truth').glob('*.csv'):
        df = pd.read_csv(fpath)
        state_name = re.search('_([a-zA-Z ]+)\.csv$', fpath.name).groups()[0]
        df['state'] = state_name
        flu_dfs.append(df)
    return pd.concat(flu_dfs, axis=0)
