"""Utilities for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from . import time_series, utils


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


def plot_model_pred_sequential(model, X, ax=None, n=1, max_len=368,
                               pad_val=-1., min_history_len=50, offset=0,
                               target_len=5, states=False, idxs=None):
    assert len(idxs) == n, f'Length of indexes provided ({len(idxs)}) does not match requested number of plots ({n}).'

    if ax is None:
        fig, axs = plt.subplots(1, n, figsize=(18, 6), sharex=True, sharey=True)
    else:
        axs = [ax]

    # Get `n` time series
    t = np.arange(min_history_len + target_len)
    if idxs is None:
        idxs = np.random.choice(len(X), size=n)
    for ax, idx in zip(axs, idxs):
        x = X[idx]
        ts = np.full((1, max_len, 1), pad_val, dtype=np.float32)
        ts[0, :min_history_len] = x[offset:offset + min_history_len]
        for j in range(target_len):
            if states:
                state_vec = np.array([[idx]])
                y_ = model.predict((ts, state_vec))
            else:
                y_ = model.predict(ts)
            ts[0, min_history_len + j, 0] = y_

        ax.plot(t[:-target_len], x[offset:offset + min_history_len], label='History')
        ax.plot(t[-target_len:], x[offset + min_history_len:offset + min_history_len + target_len], 'o', color='green', label='True target')
        ax.plot(t[-target_len:], ts[0, 1:target_len + 1, 0], 'x', color='r', label='Predictions')
        ax.legend()
        ax.set_xlabel("Time step")
        ax.set_ylabel("WILI")
        ax.set_title(f'State: {idx}')


def plot_preds_for_state(X_test, y_test, states_test, model, history_length, target_size, state, scaler=None):
    # plot the preds for target_size weeks ahead skipping every target_size inputs
    preds = model.predict(X_test[states_test == state])
    y = y_test[states_test == state]
    rmses, mean_rmse = time_series.evaluate_multiple_steps_preds(y, preds, scaler)
    if scaler != None:
        y = scaler.inverse_transform(y_test[states_test == state])
        preds = scaler.inverse_transform(preds)

    markers = np.arange(y.shape[0])[::target_size]
    multiple_step_preds = preds[::target_size, :].flatten()
    y = y[::target_size, :].flatten()
    ymarkers = multiple_step_preds[::target_size]

    print(f"RMSES for each step = {rmses}, average = {mean_rmse}")

    plt.figure(figsize=(10, 8))
    plt.plot(y, 'b-', label='True')
    plt.plot(multiple_step_preds, 'r-', label='Predicted')
    plt.plot(markers, ymarkers, '+', color='k', label='Anchor points')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('WILI')
    plt.title(f'True vs {target_size}-steps prediction for {state}')


def plot_state_predictions_multi(model, data_dict, state, target_size, scaler=None):
    state_subset = data_dict['states_test'] == state
    X_te = data_dict['X_test'][state_subset]
    yhat = model.predict(X_te)[::target_size].flatten()
    y_te = data_dict['y_test'][state_subset][::target_size].flatten()
    X_te = X_te[0].flatten()

    if scaler is not None:
        y_te = scaler.inverse_transform(y_te)
        yhat = scaler.inverse_transform(yhat)
        X_te = scaler.inverse_transform(X_te)

    t = np.arange(len(X_te) + len(y_te))
    tpre = t[:len(X_te)]
    tpost = t[-len(yhat):]

    X_te = np.append(X_te, y_te[0])
    tpre = np.append(tpre, len(tpre))

    for i in range(0, len(yhat), target_size):
        plt.plot(tpre, X_te, color='black')
        plt.plot(tpost, y_te, label='Ground truth', color='green')
        plt.plot(tpost, yhat, 'x', color='red', label='Predictions')
    plt.title(state)
