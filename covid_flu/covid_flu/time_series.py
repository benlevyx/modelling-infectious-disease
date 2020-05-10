"""Utilities for working with time series data.
"""
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from . import utils


# Defining defaults
HISTORY_SIZE = 25
TARGET_SIZE = 1


def prepare_time_series_data(ts: np.ndarray,
                             groups=None,
                             history_size=HISTORY_SIZE,
                             target_size=TARGET_SIZE,
                             pad_value=None,
                             min_history_size=1) -> tuple:
    """
    Parameters
    ----------
    ts {np.ndarray} -- The time series to be prepared
    groups {np.ndarray} -- An array of groups for each observation (optional)
    history_size {int} -- Window size for history
    target_size {int} -- Window size for prediction
    pad_value: {int} -- Optional, the value to pad with for windows that
        extend before the start of the ts. If not supplied, then the time series
        arrays will start at index `history_size`.
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
            hist, trg = _prepare_ts_single(subset, history_size, target_size, pad_value=pad_value, min_history_size=min_history_size)
            state = [group] * len(hist)
            X_all.extend(hist)
            y_all.extend(trg)
            states.extend(state)

    X_all = np.stack(X_all, axis=0)
    y_all = np.stack(y_all, axis=0)
    states = np.stack(states, axis=0)

    return X_all, y_all, states


def _prepare_ts_single(ts, history_size, target_size, pad_value=None, min_history_size=1):
    X_all, y_all = [], []
    ts_len = len(ts)
    if pad_value:
        for i in range(min_history_size, ts_len - target_size):
            hist, trg = prepare_padded_time_series(ts, history_size, target_size, target_pos=i, pad_value=pad_value)
            X_all.append(hist)
            y_all.append(trg)
    else:
        for i in range(ts_len - history_size - target_size):
            hist, trg = prepare_univariate_time_series(ts, history_size, target_size, offset=i)
            X_all.append(hist)
            y_all.append(trg)
    return X_all, y_all


def prepare_padded_time_series(X, max_history_size, target_size, target_pos=1, pad_value=-1):
    """Prepare a padded univariate time series.

    Parameters
    ----------
    X: {np.ndarray} -- The full time series
    max_history_size: {int} -- The maximum size of the history window
    target_size: {int} -- The size of the target window
    target_pos: {int} -- The position of the beginning of the target window
    pad_value: {int} -- The value to pad with (default -1)

    Returns
    -------
    'hist', 'trg'
    """
    assert target_pos + target_size <= len(X), f'Specified target position {target_pos} and target size {target_size} greater than size of X {len(X)}'
    if X.ndim == 1:
        X = np.expand_dims(X, 1)

    offset = max(0, target_pos - max_history_size)
    hist_len = target_pos - offset
    pad_len = max_history_size - hist_len

    hist = X[offset:target_pos]
    trg = X[target_pos:target_pos + target_size]

    hist = np.concatenate((hist, np.full((pad_len, 1), pad_value)), axis=0)
    return hist, trg


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


def prepare_and_split_ts(ts, groups=None, history_size=HISTORY_SIZE, target_size=TARGET_SIZE,
                         test_size=0.2, pad_value=None, min_history_size=1):
    """
    splits on train, val and test sets by state (i.e. train, val and test has instances from every state)
    :param ts:
    :param groups:
    :param history_size:
    :param target_size:
    :param test_size:
    :return:
    """
    X_all, y_all, states = prepare_time_series_data(ts, groups=groups,
                                                    history_size=history_size,
                                                    target_size=target_size,
                                                    pad_value=pad_value,
                                                    min_history_size=min_history_size)

    dct = {}
    for group in np.unique(groups):
        ts = X_all[states == group]
        y = y_all[states == group]
        len_test = int(np.ceil(len(ts) * test_size))

        offset = 0
        split_lens = [len(ts) - len_test, len_test // 2, len_test - len_test // 2]
        for split, split_len in zip(('train', 'test', 'val'), split_lens):
            ts_ = ts[offset:offset + split_len]
            y_ = y[offset:offset + split_len]
            group_ = [group] * split_len

            for name, vec in zip(('X', 'y', 'states'), (ts_, y_, group_)):
                key = f'{name}_{split}'
                curr = dct.get(key, [])
                curr.extend(vec)
                dct[key] = curr

            offset += split_len

    for key in dct:
        dct[key] = np.stack(dct[key], axis=0)
    return dct


def prepare_data(ts: np.ndarray,
                 groups=None,
                 history_size=HISTORY_SIZE,
                 target_size=TARGET_SIZE,
                 batch_size=256,
                 test_size=0.2,
                 pad_value=None,
                 min_history_size=1,
                 teacher_forcing=False,
                 states=False) -> tuple:
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
    batch_size
    test_size
    pad_value
    min_history_size
    teacher_forcing: {bool} whether to prepare the data to use teacher forcing or not.
    states: {bool} whether to include state indices in each input data or not.

    Returns
    -------

    """
    dct = prepare_and_split_ts(ts,
                               groups=groups,
                               history_size=history_size,
                               target_size=target_size,
                               test_size=test_size,
                               pad_value=pad_value,
                               min_history_size=min_history_size)

    res = []
    state2idx = {s: i for i, s in enumerate(np.unique(groups))}
    for split in ('train', 'val', 'test'):
        X, y, st = dct[f'X_{split}'], dct[f'y_{split}'], dct[f'states_{split}']
        st = np.array([state2idx[s] for s in st])
        if teacher_forcing:
            y_tf = np.zeros_like(y)
            y_tf[:, 1:, :] = y[:, :-1, :]
            y_tf[:, 0, :] = X[:, -1, :]
            if states:
                data = ((X, y_tf, st), y)
            else:
                data = ((X, y_tf), y)
        else:
            if states:
                data = ((X, st), y)
            else:
                data = (X, y)
        ds = make_dataset(data, batch_size=batch_size)
        res.append(ds)
    ds_train, ds_val, ds_test = res

    return ds_train, ds_val, ds_test, dct


def train_test_split(ts: np.ndarray, groups=None, test_size=0.2) -> tuple:
    """Perform a time series train-test split, where the training and testing
    sets do not overlap temporally.

    Parameters
    ----------
    ts -- time series to be split
    groups -- Optional. Vector identifying which elements in `ts` belong to
        which group (state)
    test_size {int} -- The test size as a proportion (0 <= test-size <= 1).
        Default 0.2

    Returns
    -------
    'ts_train', 'ts_test'
    """
    train, test = [], []
    if groups is not None:
        for group in np.unique(groups):
            ts_group = ts[np.argwhere(groups == group)]
            ts_train, ts_test = _train_test_split_single(ts_group, test_size)
            train.append(ts_train)
            test.append(ts_test)
    else:
        ts_train, ts_test = _train_test_split_single(ts, test_size)
        train.append(ts_train)
        test.append(ts_test)

    return train, test


def _train_test_split_single(ts, test_size):
    len_test = int(np.ceil(len(ts) * test_size))
    return ts[:-len_test], ts[-len_test:]


def make_dataset(data, batch_size=256, buffer_size=1000):
    """Pack data into a tensorflow Dataset.

    Parameters
    ----------
    data {tuple} -- Tuple of np.ndarrays
    batch_size {int}
    buffer_size {int}

    Returns
    -------
    'ds' {tensorflow.data.Dataset}
    """
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.cache().shuffle(buffer_size).batch(batch_size)
    return ds


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
        min_len = 1
    X, y = [], []
    for i in range(min_len, len(ts) - target_len):
        hist, trg = prepare_univariate_time_series(ts, i, target_len, offset=0)
        X.append(hist)
        y.append(trg)
    X = pad_sequences(X, maxlen=max_len, value=pad_val)
    y = np.stack(y, axis=0)
    return X, y


def prepare_all_data_walk_forward(ts_list,
                                  min_len=None,
                                  max_len=None,
                                  target_len=1,
                                  extra_info=None,
                                  pad_val=-1) -> tuple:
    """Prepare a walk-forward dataset using a list of time series.

    Parameters
    ----------
    ts_list {list} -- Iterable of np.ndarray time series
    min_len {int} -- Minimum length of a history window
    max_len {int} -- The maximum length to pad history to
    target_len {int} -- The window to be predicted
    extra_info {np.ndarray | list} -- Optional. Extra information to add to
        each time series.
    pad_val {int} -- Value to pad with (will be ignored when training model)

    Returns
    -------
    X[, extra_info], y
    """
    if not max_len:
        max_len = max([len(x) for x in ts_list])
    X, y = [], []
    extra = []
    for i, ts in enumerate(ts_list):
        x_, y_ = prepare_walk_forward_data_variable(ts,
                                                    min_len=min_len,
                                                    max_len=max_len,
                                                    target_len=target_len,
                                                    pad_val=pad_val)
        X.append(x_),
        y.append(y_)
        if extra_info is not None:
            extra.append(np.repeat(extra_info[i], len(y_)))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    if extra_info is not None:
        extra = np.concatenate(extra, axis=0)
        return (X, extra), y
    else:
        return X, y


def evaluate_multiple_steps_preds(y_true, predicted, scaler):
    # calculate the rmse for each for each of the steps individualy (i.e. error for one week ahead, 2 weeks ahead and so on)
    if scaler!=None:
        y_true = scaler.inverse_transform(y_true)
        predicted = scaler.inverse_transform(predicted)
    rmses = []
    for i in range(predicted.shape[1]):
        rmse_step = utils.calculate_rmse(y_true[:,i], predicted[:,i])
        rmses.append(rmse_step)
    rmses = np.array(rmses)
    return rmses, rmses.mean()