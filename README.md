# Models and experiments for multi-step time series forecasting.

Clone this repo in terminal using:

```
git clone https://github.com/JinlongHuang/tsf.git
```

Install required python package:

```
pip install numpy, polars, torch
```

Download data.zip from google drive, unzip it, and put the unzipped data folder into cloned tsf folder.

Data preparation and model parameters are stored in config.json.

Put definition of models in models.py.

Adjust experiments settings in exp.py.

Run experiments by running following command in the terminal:

```
python3 main.py
```
