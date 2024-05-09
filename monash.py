import dataloader
from icecream import ic
import pandas as pd
import matplotlib.pyplot as plt


DATASET = "traffic_hourly"


if __name__ == '__main__':
    (loaded_data, frequency,
     forecast_horizon,
     contain_missing_values,
     contain_equal_length) = dataloader._convert_tsf_to_dataframe(DATASET)
    ts_list = loaded_data['series_value']
    ic(loaded_data)
    ic(frequency)
    ic(forecast_horizon)
    ic(contain_missing_values)
    ic(contain_equal_length)

    ts_len = [len(ts) for ts in ts_list]
    print(ts_len)

    # for i in range(0, 10):
    #     value_series = pd.Series(ts_list[i].tolist())
    #     value_series = value_series[:500]
    #     plt.plot(value_series)

    value_series = pd.Series(ts_list[0].tolist())
    for i in range(1, len(ts_list)):
        value_series += pd.Series(ts_list[i].tolist())
    value_series /= len(ts_list)

    window_size = 24*7*10
    for start in range(0, len(value_series), window_size):
        plt.plot(value_series[start:start+window_size])
    plt.show()
