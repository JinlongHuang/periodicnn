import os
import json
from icecream import ic
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot():
    for saved_epoch in range(0, 51, 10):
        checkpoint_file = f'checkpoints/OneLinear_epoch_{saved_epoch}.pt'
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print(f"Epoch {checkpoint['epoch']:>5}, "
                  f"linear.bias: {checkpoint['model_state_dict']['linear.bias']}, "
                  f"Train Loss: {checkpoint['train_loss']:>8.3f},  "
                  f"Val Return: {- sum(checkpoint['val_loss']):>8.3f},  ")
            val_loss = pd.DataFrame(checkpoint['val_loss'])#.tail(500)
            val_preds = pd.DataFrame(checkpoint['val_preds'])#.tail(500)
            val_targets = pd.DataFrame(checkpoint['val_targets'])#.tail(500)

            model_state_dict = checkpoint['model_state_dict']
            weight_df = pd.DataFrame(
                    model_state_dict['linear.weight'].squeeze().numpy())#.tail(500)

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
            fig.add_trace(go.Bar(x=val_targets.index, y=val_targets[0],
                                 # line=dict(color='#58097d'), mode=mode,
                                 name='targets'),
                          row=1, col=1)
            fig.add_trace(go.Bar(x=val_targets.index, y=val_loss[0],
                                 # line=dict(color='#1d73bf'), mode=mode,
                                 name='loss'),
                          row=1, col=1)
            fig.add_trace(go.Bar(x=val_preds.index, y=val_preds[0],
                                 name='preds'),
                          row=2, col=1)
            fig.add_trace(go.Bar(x=weight_df.index, y=weight_df[0],
                                 name='weights'),
                          row=3, col=1)

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


def plot_pnl_heatmap():
    x_labels = _get_x_labels()
    y_labels = _get_y_labels()

    np.random.seed(42)
    f = np.random.rand(len(y_labels), len(x_labels)) / 2 + 1 / 2

    fig = go.Figure(
            data=go.Heatmap(
                z=f,
                x=x_labels,
                y=y_labels,
                text=f,
                texttemplate='%{text:.2f}',
                textfont={'size': 10},
                colorscale='Viridis',
                colorbar=dict(title='Value'),
                )
        )
    fig.show()


def _prepare_pnl_in_matrix():
    saved_epoch = 50
    checkpoint_file = f'checkpoints/OneLinear_epoch_{saved_epoch}.pt'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        print(f"Epoch {checkpoint['epoch']:>5}, "
              f"linear.bias: {checkpoint['model_state_dict']['linear.bias']}, "
              f"Train Loss: {checkpoint['train_loss']:>8.3f},  "
              f"Val Return: {- sum(checkpoint['val_loss']):>8.3f},  ")
        val_loss = pd.DataFrame(checkpoint['val_loss'])
        ic(val_loss)

    val_unix_file = 'data/val_unix.pt'
    if os.path.exists(val_unix_file):
        val_unix = torch.load(val_unix_file)
        ic(pd.DataFrame(val_unix))


def _get_x_labels() -> list[str]:
    """
    x axis labels for pnl heatmap
    """
    weekday = ['1', '2', '3', '4', '5', '6', '7']
    hour = [str(i) for i in range(0, 24, 3)]
    return [f"{w}-{h}" for w in weekday for h in hour]


def _get_y_labels() -> list[str]:
    """
    y axis labels for pnl heatmap
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
        start_unix = config['data']['train_end_unix']
        end_unix = config['data']['val_end_unix']
        timezone_offset = config['analysis']['timezone_offset']
    start_date = _unix_to_local_time(start_unix, timezone_offset)
    first_monday = (start_date
                    - timedelta(days=start_date.weekday())
                    - timedelta(hours=start_date.hour)
                    - timedelta(minutes=start_date.minute)
                    - timedelta(seconds=start_date.second)
                    )
    assert first_monday.weekday() == 0, "Variable first_monday is not Monday."
    first_monday_unix = _local_time_to_unix(first_monday)
    end_date = _unix_to_local_time(end_unix, timezone_offset)
    last_monday = (end_date
                   - timedelta(days=end_date.weekday())
                   - timedelta(hours=end_date.hour)
                   - timedelta(minutes=end_date.minute)
                   - timedelta(seconds=end_date.second)
                   )
    assert last_monday.weekday() == 0, "Variable last_monday is not Monday."
    last_monday_unix = _local_time_to_unix(last_monday)
    y = []
    for i in range(first_monday_unix, last_monday_unix + 1, 604800000):
        y.append(_unix_to_local_time(i, timezone_offset).strftime('%Y-%m-%d'))
    return y


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
    _prepare_pnl_in_matrix()
