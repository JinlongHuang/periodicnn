import os
import json
from time import time
import numpy as np
import polars as pl
import torch


def load_data() -> tuple[dict[list[tuple[torch.Tensor, torch.Tensor]]]]:
    """
    Returns:
        Each train, val and test data is a dictionary with asset names as keys.
        Asset name doesn't contain 'USDT', e.g. 'BTC' not 'BTCUSDT'.
        Each value is a list of tuples (X, Y), i.e. (batched x, batched y).

        dim of X: (batch_size, in_seq_len)
        dim of Y: (batch_size, out_seq_len)

    Note:
        Each x is a sequence of log returns:
            x_t = ln(price_t / price_{t - 1})
        Each y is a sequence of log returns:
            y_t = ln(price_t / price_{t - out_interval})

        For a sample (x, y) in a batch, x is in the past and y in the future.
        x: x_1, x_2, ... , x_{in_seq_len}
        y: y_{in_seq_len + 1}, ... , y_{in_seq_len + out_seq_len}
    """

    with open('config.json', 'r') as f:
        config = json.load(f)
        preprocess = config['data']['preprocess']
    pt_file = 'data/train_data.pt'
    if os.path.exists(pt_file) and not preprocess:
        start_time = time()
        train = torch.load('data/train_data.pt')
        val = torch.load('data/val_data.pt')
        test = torch.load('data/test_data.pt')
        print(f'Loaded bateched data in {time() - start_time:.2f} seconds')
        return train, val, test
    else:
        start_time = time()
        print('Preprocessing data...')
        _save_batched_data()
        print(f'Preprocessed and saved batched data in \
                {time() - start_time:.2f} seconds')
        with open('config.json', 'r') as f:
            config = json.load(f)
            config['data']['preprocess'] = False
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        return load_data()
###
def _save_batched_data():
    with open('config.json', 'r') as f:
        config = json.load(f)
        binarize = config['data']['binarize']
        batch_size = config['data']['batch_size']
        in_seq_len = config['data']['in_seq_len']
        out_seq_len = config['data']['out_seq_len']
        in_interval = config['data']['in_interval']
        out_interval = config['data']['out_interval']
        in_len = in_seq_len * in_interval
        out_len = out_seq_len * out_interval
        train_end_unix = config['data']['train_end_unix']
        val_end_unix = config['data']['val_end_unix']
        assets_for_preprocess = config['data']['assets_for_preprocess']

    train_data = dict()
    train_unix = dict()
    val_data = dict()
    val_unix = dict()
    test_data = dict()
    for entry in os.scandir('data/'):
        asset = entry.name[:-4]
        if not entry.is_dir() or (asset not in assets_for_preprocess):
            continue
        raw_file = f'{entry.path}/raw.parquet'

        df = pl.read_parquet(raw_file)
        if df.filter(df['close'] <= 0).height > 0:
            raise ValueError("Close price must be positive")

        # Calculate log returns
        df = df.with_columns(
                np.log(df['close'] / df['close'].shift(
                    n=in_interval, fill_value=1)
                       ).alias('log_ret_x'))
        df = df.with_columns(
                np.log(df['close'] / df['close'].shift(
                    n=out_interval, fill_value=1)
                       ).alias('log_ret_y'))
        df = df.tail(df.height
                     - max(in_interval, out_interval))  # remove filled values
        df = df.drop('close')

        samples = []
        train_data[asset] = []
        train_unix[asset] = []
        val_data[asset] = []
        val_unix[asset] = []
        test_data[asset] = []
        # Note: i loops through index instead of unix
        # Due to the possibility of missing data, unix time diff >= 1 minute
        for i in range(0, df.height - in_len - out_len, out_len):
            x_start_unix = df.slice(i, 1).select('unix').to_numpy()[0][0]
            y_end_unix = x_start_unix + (in_len + out_len) * 60 * 1000
            removable = (
                    (x_start_unix < train_end_unix
                     and train_end_unix <= y_end_unix < val_end_unix)
                    or (train_end_unix <= x_start_unix < val_end_unix
                        and val_end_unix <= y_end_unix)
                    )
            if removable:
                samples = []
                continue

            x = df.gather_every(in_interval, i).head(in_seq_len)
            x = x.select(['log_ret_x']).to_numpy()
            x = torch.tensor(x, dtype=torch.float32).squeeze(1)

            y = df.gather_every(out_interval, i + in_len + out_len
                                ).head(out_seq_len)
            y = y.select('log_ret_y').to_numpy()
            y = torch.tensor(y, dtype=torch.float32).squeeze(1)

            if binarize:
                x, y = _binarize_data(x, y)

            if len(x) != in_seq_len or len(y) != out_seq_len:
                continue
            samples.append((x, y))

            if len(samples) == batch_size:
                X, Y = zip(*samples)
                X = torch.stack(X)
                Y = torch.stack(Y)

                if y_end_unix < train_end_unix:
                    train_data[asset].append((X, Y))
                elif (y_end_unix < val_end_unix
                      and x_start_unix >= train_end_unix):
                    val_data[asset].append((X, Y))
                elif x_start_unix >= val_end_unix:
                    test_data[asset].append((X, Y))

                samples = []

            y_start_unix = x_start_unix + in_len * 60 * 1000
            if y_end_unix < train_end_unix:
                train_unix[asset].append(y_start_unix)
            elif y_end_unix < val_end_unix and x_start_unix >= train_end_unix:
                val_unix[asset].append(y_start_unix)

        print(f'{asset:<6}',
              f'train n_batch: {len(train_data[asset]):>3}     ',
              f'val n_batch: {len(val_data[asset]):>3}     ',
              f'test n_batch: {len(test_data[asset]):>3}')
        torch.save(train_unix[asset][:len(train_data[asset])*batch_size],
                   'data/train_unix.pt')
        torch.save(val_unix[asset][:len(val_data[asset])*batch_size],
                   'data/val_unix.pt')
    torch.save(train_data, 'data/train_data.pt')
    torch.save(val_data, 'data/val_data.pt')
    torch.save(test_data, 'data/test_data.pt')


def _binarize_data(x: torch.Tensor, y: torch.Tensor):
    """
    output: x, y are torch.Tensor with entries 1.0 or -1.0
    """
    x = (x + 0.00000001).sign()
    y = (y + 0.00000001).sign()

    contains_zero = (x == 0).any() or (y == 0).any()
    if contains_zero.item():
        raise ValueError("Binarized data contains zero.")
    return x, y
