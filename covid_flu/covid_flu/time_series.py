"""Utilities for working with time series data.
"""
from typing import Union

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Defining defaults
HISTORY_SIZE = 25
TARGET_SIZE = 1


def prepare_univariate_time_series(X, input_window, pred_window, offset=0):
    """Prepare a single univariate time series.

    Arguments:
    ----------
    X {np.ndarray} -- The input time series
    input_window {int} -- The window size for training data
    pred_window {int}  -- The window size for prediction

    Keyword arguments:
    ------------------
    offset {int} -- Where to start the time series

    Returns:
    --------
    'X_input', 'X_pred'
    """
    assert len(X) >= offset + input_window + pred_window, \
        f'''Length of time series {len(X)} less than combined
        length of offset {offset}, input_window {input_window}
        and pred_window {pred_window}'''

    if len(X.shape) == 1:
        X = np.expand_dims(X, 1)
    X_input = X[offset:offset + input_window]
    X_pred = X[offset + input_window:offset + input_window + pred_window]

    return X_input, X_pred


def prepare_time_series_data(ts: np.ndarray,
                             groups=None,
                             history_size=HISTORY_SIZE,
                             target_size=TARGET_SIZE) -> tuple:
    """

    Parameters
    ----------
    ts {np.ndarray} -- The time series to be prepared
    groups {np.ndarray} -- An array of groups for each observation (optional)
    history_size {int} -- Window size for history
    target_size {int} -- Window size for prediction

    Returns
    -------
    'X_all', 'y_all' {tuple(np.ndarray)} -- History and prediction arrays
    """
    X_all, y_all, states = [], [], []
    if groups is None:
        X_all, y_all = _prepare_ts_single(ts, history_size, target_size)
    else:
        for group in np.unique(groups):
            subset = ts[np.argwhere(groups == group)]
            subset_states = groups[np.argwhere(groups == group)]
            hist, trg = _prepare_ts_single(subset, history_size, target_size)
            state = [group]*len(hist)
            X_all.extend(hist)
            y_all.extend(trg)
            states.extend(state)

    X_all = np.stack(X_all, axis=0)
    y_all = np.stack(y_all, axis=0)
    states = np.stack(states, axis=0)

    return X_all, y_all, states


def _prepare_ts_single(ts, history_size, target_size):
    X_all, y_all = [], []
    ts_len = len(ts)
    for i in range(ts_len - history_size - target_size):
        hist, trg = prepare_univariate_time_series(ts, history_size, target_size, offset=i)
        X_all.append(hist)
        y_all.append(trg)
    return X_all, y_all


def prepare_and_split_ts(ts, groups=None, history_size=HISTORY_SIZE, target_size=TARGET_SIZE,
                         test_size=0.2):
    """
    splits on train, val and test sets by state (i.e. train, val and test has instances from every state)
    :param ts:
    :param groups:
    :param history_size:
    :param target_size:
    :param test_size:
    :return:
    """
    X_all, y_all, states = prepare_time_series_data(ts, groups=groups, history_size=history_size, target_size=target_size)
    X_train = []
    y_train = []
    states_train = []
    X_val = []
    y_val = []
    states_val = []
    X_test = []
    y_test = []
    states_test = []
    for group in np.unique(groups):
        ts = X_all[states == group]
        y = y_all[states == group]
        len_test = int(np.ceil(len(ts) * test_size))
        ts_train = ts[:-len_test]
        state_train = [group] * len(ts_train)
        y_state_train = y[:-len_test]
        ts_val = ts[-len_test:-len_test + len_test // 2]
        state_val = [group] * len(ts_val)
        y_state_val = y[-len_test:-len_test + len_test // 2]
        ts_test = ts[-len_test + len_test // 2:]
        y_state_test = y[-len_test + len_test // 2:]
        state_test = [group] * len(ts_test)
        X_train.extend(ts_train)
        X_val.extend(ts_val)
        X_test.extend(ts_test)
        y_train.extend(y_state_train)
        y_val.extend(y_state_val)
        y_test.extend(y_state_test)
        states_train.extend(state_train)
        states_val.extend(state_val)
        states_test.extend(state_test)

    dict = {}
    dict['X_train'] = np.stack(X_train, axis=0)
    dict['X_val'] = np.stack(X_val, axis=0)
    dict['X_test'] = np.stack(X_test, axis=0)
    dict['y_train'] = np.stack(y_train, axis=0)
    dict['y_val'] = np.stack(y_val, axis=0)
    dict['y_test'] = np.stack(y_test, axis=0)
    dict['states_train'] = np.stack(states_train, axis=0)
    dict['states_val'] = np.stack(states_val, axis=0)
    dict['states_test'] = np.stack(states_test, axis=0)
    return dict

def make_datasets(X_train, X_val,  X_test, y_train, y_val, y_test, batch_size=256, buffer_size=1000):
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.cache().shuffle(buffer_size).batch(batch_size)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.cache().shuffle(buffer_size).batch(batch_size)

    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ds_test = ds_test.cache().shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_val, ds_test


def prepare_data(ts: np.ndarray,
                 groups=None,
                 history_size=HISTORY_SIZE,
                 target_size=TARGET_SIZE,
                 batch_size=256,
                 test_size=0.2) -> tuple:
    """
    Prepare time series data into history and target windows, splitting into
    train val and test, batched, repeating tensorflow datasets. returns a dict with
    to aid plotting and evaluation later.

    Parameters
    ----------
    ts
    groups
    history_size
    target_size
    test_size

    Returns
    -------
    'ds_train', 'ds_test'
    """
    dict = prepare_and_split_ts(ts,
                                  groups=groups,
                                  history_size=history_size,
                                  target_size=target_size,
                                  test_size=test_size)
    X_train = dict['X_train']
    X_val = dict['X_val']
    X_test = dict['X_test']
    y_train = dict['y_train']
    y_val = dict['y_val']
    y_test = dict['y_test']
    ds_train, ds_val, ds_test = make_datasets(X_train, X_val,  X_test, y_train, y_val, y_test, batch_size=batch_size)

    return ds_train, ds_val, ds_test, dict


def walk_forward_val(model, ts,
                     history_size=HISTORY_SIZE,
                     target_size=TARGET_SIZE,
                     return_count=False) -> Union[float, tuple]:
    """
    Conduct walk-forward validation of this model on the entire provided time series
    Parameters
    ----------
    model -- The model to be evaluated
    ts {np.ndarray} -- The evaluation time series
    history_size {int} -- The window to use for model input
    target_size {int} -- The target prediction window size
    return_count {bool} -- Whether to return the number of prediction steps
        used in the validation

    Returns
    -------
    'score' {float} -- The validated score for this model.
    """
    ts_len = len(ts)
    if ts.ndim == 1:
        ts = np.expand_dims(ts, 1)
    window_len = history_size + target_size
    Xs = np.zeros((ts_len - window_len, history_size, ts.shape[1]))
    ys = np.zeros((ts_len - window_len, target_size, ts.shape[1]))

    for offset in range(ts_len - window_len):
        Xs[offset] = ts[offset:offset + history_size]
        ys[offset] = ts[offset + history_size:offset + window_len]

    y_preds = model.predict(Xs)
    if y_preds.ndim != 3:
        y_preds = np.expand_dims(y_preds, 2)
    mse = np.mean((ys - y_preds) ** 2).flatten()[0]
    if return_count:
        return mse, len(ys)
    else:
        return mse


def walk_forward_val_multiple(model, ts_list,
                              history_size=HISTORY_SIZE,
                              target_size=TARGET_SIZE) -> float:
    """
    Conduct walk-forward validation for all states, average the results.

    Parameters
    ----------
    model -- The model to be validated
    ts_list {list | np.ndarray} -- Array of time series vector
    history_size {int} -- The window to use for model input
    target_size {int} -- The target prediction window size

    Returns
    -------
    'mse' {float} -- The weighted average MSE across all the states (weighted
        by length of time series)
    """
    total_error = 0.
    total_steps = 0
    for ts in ts_list:
        mse_state, n_preds = walk_forward_val(model, ts,
                                              history_size=history_size,
                                              target_size=target_size,
                                              return_count=True)

        total_error += mse_state * n_preds
        total_steps += n_preds
    return total_error / total_steps


def prepare_walk_forward_data_variable(ts, min_len=None, max_len=None, target_len=1, pad_val=-1) -> tuple:
    """Process time series `ts` into either an array of padded history
    sequences and output targets of length `target_len`.

    Parameters
    ----------
    ts {np.ndarray} -- The time series to be processed
    max_len {int} -- The maximum length to pad to
    target_len {int} -- The size of the targets (prediction window)
    pad_val {int | float} -- The value to pad with (default -1)

    Returns
    -------
    X, y
    """
    if not min_len:
        min_len = 0
    X, y = [], []
    for i in range(1, len(ts)):
        hist, trg = prepare_univariate_time_series(ts, i, target_len, min_len)
        X.append(hist)
        y.append(trg)
    X = pad_sequences(X, maxlen=max_len, value=pad_val)
    y = np.stack(y, axis=0)
    return X, y


def prepare_all_data_walk_forward(ts_list, min_len=None, max_len=None, target_len=1, pad_val=-1) -> tuple:
    if not max_len:
        max_len = max([len(x) for x in ts_list])
    X, y = [], []
    for ts in ts_list:
        x_, y_ = prepare_walk_forward_data_variable(ts,
                                                    min_len=min_len,
                                                    max_len=max_len,
                                                    target_len=target_len,
                                                    pad_val=pad_val)
        X.append(x_),
        y.append(y_)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y
