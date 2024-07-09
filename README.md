# Multi-step time series forecasting using neural network with periodic activation function

Clone this repo in terminal using:

```
git clone https://github.com/JinlongHuang/periodicnn.git
```

Install required python package:

```
pip install numpy, polars, torch
```

Put the data folder into cloned periodicnn folder.

Data preparation and model parameters are stored in config.json.

Run experiments by running following command in the terminal:

```
python3 main.py
```

To use WandB, create a wandb_api.env file in the root directory.
wandb_api.env file contains a single line:
WANDB_API=YOUR_WANDB_API_KEY_FROM_THEIR_WEBSITE
