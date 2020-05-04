"""Utilities for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt


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
