"""Utilities for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from . import time_series


def plot_history(history, ax=None):
    """Plot a tensorflow history.
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(history.history['loss'], label='Training')
    ax.plot(history.history['val_loss'], label='Validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()


def plot_time_series_prediction(X, y_true, y_pred, ax=None):
    """Plot the original time series and the predicted next point(s),
    compared to the true next points.

    Arguments:
    ----------
    X {np.ndarray} -- The data the model has seen
    y_true {np.ndarray} -- The true next points(s)
    y_pred {np.ndarray} -- The predicted next point(s)
    """
    X = X.ravel()
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    assert len(y_true) == len(y_pred), f'True and predicted data not same length ({len(y_true)}, {len(y_pred)})'

    if ax is None:
        ax = plt.gca()

    time_X = np.arange(len(X))
    time_y = np.arange(len(X), len(X) + len(y_true))

    ax.plot(time_X, X, label='History')
    ax.plot(time_y, y_true, 'o', c='green', label='Ground truth')
    ax.plot(time_y, y_pred, 'x', c='r', label='Model predictions')

    ax.set_xlabel('Time step')
    ax.legend()


def plot_model_pred(model, X, y=None, idx=None, ax=None, n=1):
    """
    Plot time series model predictions.

    Parameters
    ----------
    model -- The model to plot
    X {np.ndarray | tf.data.Dataset} -- Either the history windows or a
        tensorflow dataset
    y {None} -- Optional. The true prediction windows. If None, then X
        must be a tensorflow dataset.
    idx {int} -- Optional. The index of the time series to plot. X and y must
        be numpy ndarray's.
    ax {matplotlib.axes.Axis} -- Optional. The axis to plot on.

    Returns
    -------
    None
    """
    if ax is not None:
        axs = [ax]
    else:
        fig, axs = plt.subplots(1, n, figsize=(18, 6))

    for ax in axs:
        if y is None and isinstance(X, tf.data.Dataset):
            X_eval, y_eval = next(X.unbatch().batch(1).take(1).as_numpy_iterator())
        else:
            if not idx:
                idx = np.random.choice(len(X))

            X_eval = X[idx].reshape(1, -1, 1)
            y_eval = y[idx].reshape(-1, 1)
        y_pred = model.predict(X_eval).reshape(-1, 1)
        plot_time_series_prediction(X_eval, y_eval, y_pred, ax=ax)


def plot_model_pred_padded(model, X, ax=None, n=1, max_len=368):
    if ax is None:
        axs = plt.subplots(1, n, figsize=(18, 6))
    else:
        axs = [ax]

    # Get `n` time series
    xs = X[np.random.choice(len(X), size=n)]
    for i in range(n):
        x, y_true = time_series.prepare_walk_forward_data_variable(xs[i],
                                                                max_len=max_len,
                                                                target_len=1)
        y_pred = model.predict(x)
