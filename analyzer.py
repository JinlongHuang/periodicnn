import os
import json
from icecream import ic
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import polars as pl
import torch
import plotly.graph_objects as go

from dataloader import load_data


PLOT_INTERVAL = 1  # hours
SAVED_EPOCH = 50


def plot_bar():
    for saved_epoc in range(0, 301, 50):
        checkpoint_file = f'checkpoints/OneLinear_epoch_{saved_epoc}.pt'
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            model_state_dict = checkpoint['model_state_dict']
            weight_df = pd.DataFrame(
                    model_state_dict['linear.weight'].squeeze().numpy())

            fig = go.Figure([go.Bar(x=weight_df.index, y=weight_df[0])
                             ])

            grid_color = "#151515"
            fig.update_layout(
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    margin=dict(t=0, l=0, b=0, r=0),
                    autosize=True,
                    font=dict(family="Courier New, monospace",
                              size=25, color="grey"),
                    barmode='relative',
                    # height=3000,
                    )
            fig.update_yaxes(showgrid=True, gridwidth=1,
                             gridcolor=grid_color, linecolor=grid_color,
                             # zeroline=False
                             )

            fig.show()


def plot_weight_heatmap(checkpoint_file, params_name):
    """
    Plot the weight matrix as a Heatmap

    Args:
        checkpoint_file: str
            The file path of the checkpoint file
        params_name: str
            The name of the parameter to plot

    Example:
    plot_weight_heatmap('checkpoints/OneLinear_epoch_0.pt', 'linear.weight')
    """
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f'{checkpoint_file} not found')

    checkpoint = torch.load(checkpoint_file)
    model_state_dict = checkpoint['model_state_dict']
    params_df = pl.DataFrame(
            model_state_dict[params_name].squeeze().numpy())

    n_rows = params_df.shape[0]
    n_cols = params_df.shape[1]
    x_labels = np.arange(n_cols)
    y_labels = np.arange(n_rows)
    z = params_df.to_numpy()

    fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale='Viridis',
                colorbar=dict(title='Weight'),
                )
        )
    fig.show()


def plot_val_line(checkpoint_file: str):
    ts_name = 'traffic_hourly'
    ts_index = -1
    batch_index = 1
    sample_index = 3

    # get input and target
    _, val_data, _, _ = load_data()
    X, Y = val_data[ts_name][ts_index][batch_index]
    input = X[sample_index].squeeze().numpy()
    target = Y[sample_index].squeeze().numpy()

    # get pred
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f'{checkpoint_file} not found')
    checkpoint = torch.load(checkpoint_file)
    val_preds = checkpoint['val_preds']
    pred = val_preds[batch_index][sample_index]

    # define x labels
    in_seq_len = input.shape[0]
    out_seq_len = target.shape[0]
    input_labels = np.arange(in_seq_len)
    target_labels = np.arange(in_seq_len, in_seq_len + out_seq_len)

    # plot
    fig = go.Figure()
    blue = '#2d86ba'
    green = '#2bc459'
    fig.add_trace(go.Scatter(x=input_labels, y=input, mode='lines',
                             line={'color': blue},
                             name='input'))
    fig.add_trace(go.Scatter(x=target_labels, y=target, mode='lines',
                             line={'color': blue,
                                   'dash': 'dash'},
                             name='target'))
    fig.add_trace(go.Scatter(x=target_labels, y=pred, mode='lines',
                             line={'color': green},
                             name='pred'))

    fig.show()


def plot_pnl_heatmap():
    x_labels = _get_x_labels()
    y_labels = _get_y_labels()
    pnl = _prepare_pnl_as_matrix()
    pred = np.around(_prepare_pred_as_matrix(), decimals=2).astype(str)
    target = np.around(_prepare_target_as_matrix(), decimals=2).astype(str)

    fig = go.Figure(
            data=go.Heatmap(
                z=pnl,
                x=x_labels,
                y=y_labels,
                text=np.char.add(np.char.add(np.char.add('pred: ', pred),
                                             '   log return: '), target),
                # texttemplate='pred: %{text:.2f}',
                # textfont={'size': 10, 'color': 'red'},
                colorscale='Viridis',
                colorbar=dict(title='PnL'),
                )
        )
    fig.show()


def _prepare_target_as_matrix() -> np.ndarray:
    checkpoint_file = f'checkpoints/OneLinear_epoch_{SAVED_EPOCH}.pt'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        val_target = checkpoint['val_targets']
        val_target_plotted = [sum(val_target[i:i + PLOT_INTERVAL])
                              for i in range(0, len(val_target), PLOT_INTERVAL)
                              ][:-1]
    return _pad_zeros_to_plot(val_target_plotted)


def _prepare_pred_as_matrix() -> np.ndarray:
    checkpoint_file = f'checkpoints/OneLinear_epoch_{SAVED_EPOCH}.pt'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        val_pred = checkpoint['val_preds']
        val_pred_plotted = [sum(val_pred[i:i + PLOT_INTERVAL]) / PLOT_INTERVAL
                            for i in range(0, len(val_pred), PLOT_INTERVAL)
                            ][:-1]
    return _pad_zeros_to_plot(val_pred_plotted)


def _prepare_pnl_as_matrix() -> np.ndarray:
    checkpoint_file = f'checkpoints/OneLinear_epoch_{SAVED_EPOCH}.pt'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        val_pnl = checkpoint['val_pnl']
        val_pnl_plotted = [sum(val_pnl[i:i + PLOT_INTERVAL])
                           for i in range(0, len(val_pnl), PLOT_INTERVAL)
                           ][:-1]
    return _pad_zeros_to_plot(val_pnl_plotted)


def _pad_zeros_to_plot(plotted_data: list) -> np.ndarray:
    val_unix_file = 'data/crypto/val_unix.pt'
    if os.path.exists(val_unix_file):
        val_unix = torch.load(val_unix_file)
    with open('config.json', 'r') as f:
        config = json.load(f)
        start_unix = config['data']['crypto_train_end_unix']
        timezone_offset = config['analysis']['timezone_offset']

    val_start_date = _unix_to_local_time(val_unix[0], timezone_offset)
    plot_start_date = _unix_to_local_time(start_unix, timezone_offset)
    first_monday = _previous_monday(plot_start_date)
    val_start_index = int(
            (val_start_date - first_monday).total_seconds() / 3600
            // PLOT_INTERVAL)

    result = np.zeros(len(_get_y_labels()) * len(_get_x_labels()))
    result[val_start_index:
           val_start_index + len(plotted_data)] = plotted_data
    result = result.reshape(len(_get_y_labels()), len(_get_x_labels()))
    return result


def _get_x_labels() -> list[str]:
    """
    x axis labels for pnl heatmap
    """
    weekday = ['1', '2', '3', '4', '5', '6', '7']
    hour = [str(i) for i in range(0, 24, )]
    return [f"{w}-{h}" for w in weekday for h in hour]


def _get_y_labels() -> list[str]:
    """
    y axis labels for pnl heatmap
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
        start_unix = config['data']['crypto_train_end_unix']
        end_unix = config['data']['crypto_val_end_unix']
        timezone_offset = config['analysis']['timezone_offset']
    start_date = _unix_to_local_time(start_unix, timezone_offset)
    first_monday = _previous_monday(start_date)
    first_monday_unix = _local_time_to_unix(first_monday)

    end_date = _unix_to_local_time(end_unix, timezone_offset)
    last_monday = _previous_monday(end_date)
    last_monday_unix = _local_time_to_unix(last_monday)
    y = []
    for i in range(first_monday_unix, last_monday_unix + 1, 604800000):
        y.append(_unix_to_local_time(i, timezone_offset).strftime('%Y-%m-%d'))
    return y


def _previous_monday(date: datetime) -> datetime:
    return (date
            - timedelta(days=date.weekday())
            - timedelta(hours=date.hour)
            - timedelta(minutes=date.minute)
            - timedelta(seconds=date.second)
            )


def _unix_to_local_time(unix: int, timezone_offset: int) -> datetime:
    """
    Unit for unix could be in seconds or milliseconds
    """
    if len(str(unix)) == 10:
        unix *= 1000
    return datetime.fromtimestamp(unix / 1000,
                                  timezone(timedelta(hours=timezone_offset)))


def _local_time_to_unix(local_time: datetime) -> int:
    return int(local_time.timestamp() * 1000)


if __name__ == '__main__':
    # plot_pnl_heatmap()
    # plot_weight_heatmap('checkpoints/AdaFunc_epoch_0.pt', 'linear.weight')
    plot_val_line('checkpoints/AdaFunc_epoch_0.pt')
