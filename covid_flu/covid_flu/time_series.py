"""Utilities for working with time series data.
"""
from typing import Union

import numpy as np
import pandas as pd


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
    assert len(X) > offset + input_window + pred_window, \
        f'''Length of time series {len(X)} less than combined
        length of offset {offset}, input_window {input_window}
        and pred_window {pred_window}'''

    if len(X.shape) == 1:
        X = np.expand_dims(X, 1)
    X_input = X[offset:offset + input_window]
    X_pred = X[offset + input_window:offset + input_window + pred_window]

    return X_input, X_pred
