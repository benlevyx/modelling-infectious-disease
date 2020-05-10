from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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
    flu_data = pd.read_csv(config.processed / 'flu_ground_truth_imputed.csv')
    return flu_data

def load_covid_data():
    covid_data = pd.read_csv(config.raw / 'Covid_data.csv', index_col=0)
    covid_data = covid_data.melt(id_vars=['date'])
    covid_data.columns = ['date', 'state', 'cases']
    covid_data = covid_data.fillna(0.)
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    # Switching to new cases
    covid_data['total_cases'] = covid_data['cases']
    covid_data['cases'] = covid_data['total_cases'] - covid_data['total_cases'].shift(1, fill_value=0)
    return covid_data


def load_state_data():
    state_data = pd.read_csv(config.state_stats / 'state_stats.csv')
    latlon = pd.read_csv(config.state_stats / 'statelatlong.csv')[['State', 'City']]
    state_data = pd.merge(state_data, latlon, left_index=True, right_on='State')
    state_data = state_data.rename(columns={'City': 'state'})
    return state_data


def scale_data(x: np.ndarray, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    x_sc = scaler.fit_transform(x.reshape(-1, 1)).flatten()
    return x_sc


def calc_rmse_model(y_true, x, model, scaler=None):
    preds = model.predict(x)
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.flatten())
        preds = scaler.inverse_transform(preds.flatten())
    return calculate_rmse(y_true, preds)


def calculate_rmse(y_true, y_pred):
    # mse = mean_squared_error(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def save_weights(model, name):
    model.training_network.save_weights(str(config.models / f'{name}.h5'))


def load_weights(model, name):
    model.training_network.load_weights(str(config.models / f'{name}.h5'))
