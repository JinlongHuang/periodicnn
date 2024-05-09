import os
import json
from time import time
from datetime import datetime
from distutils.util import strtobool

import numpy as np
import pandas as pd
import polars as pl
import torch


def load_data() -> tuple[dict[list[tuple[torch.Tensor, torch.Tensor]]]]:
    with open('config.json', 'r') as f:
        config = json.load(f)
        dataset_category = config['data']['dataset_category']

    if dataset_category == 'monash':
        return _load_monash_data()
    elif dataset_category == 'crypto':
        return _load_crypto_data()


def _load_crypto_data():
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
    pt_file = 'data/crypto/train_data.pt'
    if os.path.exists(pt_file) and not preprocess:
        start_time = time()
        train = torch.load('data/crypto/train_data.pt')
        val = torch.load('data/crypto/val_data.pt')
        test = torch.load('data/crypto/test_data.pt')
        print(f'Loaded bateched crypto data in '
              f'{time() - start_time:.2f} seconds')
        return train, val, test
    else:
        start_time = time()
        print('Preprocessing crypto data...')
        _save_batched_crypto_data()
        print(f'Preprocessed and saved batched crypto data in \
                {time() - start_time:.2f} seconds')
        with open('config.json', 'r') as f:
            config = json.load(f)
            config['data']['preprocess'] = False
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        return _load_crypto_data()


def _save_batched_crypto_data():
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
        train_end_unix = config['data']['crypto_train_end_unix']
        val_end_unix = config['data']['crypto_val_end_unix']
        assets_for_preprocess = config['data']['ts_for_preprocess']

    train_data = dict()
    train_unix = dict()
    val_data = dict()
    val_unix = dict()
    test_data = dict()
    for entry in os.scandir('data/crypto/'):
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
                   'data/crypto/train_unix.pt')
        torch.save(val_unix[asset][:len(val_data[asset])*batch_size],
                   'data/crypto/val_unix.pt')
    torch.save(train_data, 'data/crypto/train_data.pt')
    torch.save(val_data, 'data/crypto/val_data.pt')
    torch.save(test_data, 'data/crypto/test_data.pt')


def _load_monash_data():
    with open('config.json', 'r') as f:
        config = json.load(f)
        preprocess = config['data']['preprocess']
    pt_file = 'data/monash/train_data.pt'
    if os.path.exists(pt_file) and not preprocess:
        start_time = time()
        train = torch.load('data/monash/train_data.pt')
        val = torch.load('data/monash/val_data.pt')
        test = torch.load('data/monash/test_data.pt')
        zero_shot = torch.load('data/monash/zero_shot_data.pt')
        print(f'Loaded bateched monash data in '
              f'{time() - start_time:.2f} seconds')
        return train, val, test, zero_shot
    else:
        start_time = time()
        print('Preprocessing monash data...')
        _save_batched_monash_data()
        print(f'Preprocessed and saved batched monash data in '
              f'{time() - start_time:.2f} seconds')
        with open('config.json', 'r') as f:
            config = json.load(f)
            config['data']['preprocess'] = False
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        return _load_monash_data()


def _save_batched_monash_data():
    """
    Returns:
        Each train, val and test data is a dictionary with ts name as keys.
        Each value is a list of tuples (X, Y), i.e. (batched x, batched y).

        dim of X: (batch_size, in_seq_len)
        dim of Y: (batch_size, out_seq_len)
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
        batch_size = config['data']['batch_size']
        in_seq_len = config['data']['in_seq_len']
        out_seq_len = config['data']['out_seq_len']
        in_interval = config['data']['in_interval']
        out_interval = config['data']['out_interval']
        in_len = in_seq_len * in_interval
        out_len = out_seq_len * out_interval
        step_size = config['data']['step_size']
        zero_shot_percentage = config['data']['monash_zero_shot_percentage']
        train_percentage = config['data']['monash_train_percentage']
        val_percentage = config['data']['monash_val_percentage']
        ts_for_preprocess = config['data']['ts_for_preprocess']

    train_data = dict()
    val_data = dict()
    test_data = dict()
    zero_shot_data = dict()
    for ts_name in ts_for_preprocess:
        (loaded_data, frequency,
         forecast_horizon,
         contain_missing_values,
         contain_equal_length) = _convert_tsf_to_dataframe(ts_name)
        ts_list = loaded_data['series_value']
        num_ts = len(ts_list)
        trained_ts_list = ts_list[:int(num_ts * (1 - zero_shot_percentage))]
        zero_shot_ts_list = ts_list[int(num_ts * (1 - zero_shot_percentage)):]

        train_data[ts_name] = []  # a list of list:train_batches_per_ts
        val_data[ts_name] = []  # a list of list:val_batches_per_ts
        test_data[ts_name] = []  # a list of list:test_batches_per_ts
        for ts in trained_ts_list:
            value_series = pd.Series(ts).to_numpy()
            train_end_index = int(len(value_series) * train_percentage)
            val_end_index = int(len(value_series)
                                * (train_percentage + val_percentage))

            train_batches_per_ts = []
            val_batches_per_ts = []
            test_batches_per_ts = []
            samples = []
            for i in range(0, len(value_series) - in_len - out_len, step_size):
                sample_end_index = i + in_len + out_len
                removable = (
                    (i < train_end_index
                     and train_end_index <= sample_end_index < val_end_index)
                    or (train_end_index <= i < val_end_index
                        and val_end_index <= sample_end_index)
                )
                if removable:
                    samples = []
                    continue

                x = torch.Tensor(value_series[i:i + in_len])
                y = torch.Tensor(value_series[i + in_len:i + in_len + out_len])
                if len(x) != in_len or len(y) != out_len:
                    continue
                samples.append((x, y))

                if len(samples) == batch_size:
                    X, Y = zip(*samples)
                    X = torch.stack(X)
                    Y = torch.stack(Y)

                    if sample_end_index < train_end_index:
                        train_batches_per_ts.append((X, Y))
                    elif (i >= train_end_index
                          and sample_end_index < val_end_index):
                        val_batches_per_ts.append((X, Y))
                    elif sample_end_index >= val_end_index:
                        test_batches_per_ts.append((X, Y))

                    samples = []

            train_data[ts_name].append(train_batches_per_ts)
            val_data[ts_name].append(val_batches_per_ts)
            test_data[ts_name].append(test_batches_per_ts)

            print(f'{ts_name:<30}',
                  f'train n_batch: {len(train_batches_per_ts):>3}     ',
                  f'val n_batch: {len(val_batches_per_ts):>3}     ',
                  f'test n_batch: {len(test_batches_per_ts):>3}')

        zero_shot_data[ts_name] = []
        for ts in zero_shot_ts_list:
            value_series = pd.Series(ts).to_numpy()

            zero_shot_batches_per_ts = []
            samples = []
            for i in range(0, len(value_series) - in_len - out_len, step_size):
                x = torch.Tensor(value_series[i:i + in_len])
                y = torch.Tensor(value_series[i + in_len:i + in_len + out_len])
                if len(x) != in_len or len(y) != out_len:
                    continue
                samples.append((x, y))

                if len(samples) == batch_size:
                    X, Y = zip(*samples)
                    X = torch.stack(X)
                    Y = torch.stack(Y)

                    zero_shot_batches_per_ts.append((X, Y))

                    samples = []
            zero_shot_data[ts_name].append(zero_shot_batches_per_ts)
            print(f'{ts_name:<30}',
                  f'zero_shot n_batch: {len(zero_shot_batches_per_ts):>3}')

    torch.save(train_data, 'data/monash/train_data.pt')
    torch.save(val_data, 'data/monash/val_data.pt')
    torch.save(test_data, 'data/monash/test_data.pt')
    torch.save(zero_shot_data, 'data/monash/zero_shot_data.pt')


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


# for monash data
# copied from github.com/rakshitha123/TSForecasting
def _convert_tsf_to_dataframe(
    time_series_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    full_file_path_and_name = 'data/monash/' + time_series_name + '.tsf'
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


# _load_monash_data()
