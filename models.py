import json
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


with open('config.json', 'r') as f:
    config = json.load(f)
    seed = config['train']['torch_seed']
    torch.manual_seed(seed)


class OneLinear(nn.Module):
    def __init__(self, in_seq_len, out_seq_len):
        """
        in_seq_len: input sequence length
        out_seq_len: output sequence length
        """
        super().__init__()
        # self.dropout = nn.Dropout(p=0.2)

        self.linear = nn.Linear(in_seq_len, out_seq_len)
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        # uncomment next line to use custom initialization
        # self._init_weight_bias_linear_increasing(in_seq_len)

        # self.layer_norm = nn.LayerNorm(in_seq_len, elementwise_affine=False)
        # self.batch_norm = nn.BatchNorm1d(in_seq_len, affine=False)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        shape of input : (batch_size, in_seq_len)
        shape of output: (batch_size, out_seq_len)
        """
        # x = self.dropout(x)
        norm = 0
        if norm == 0:
            return self.tanh(self.linear(x))
        elif norm == 1:
            return self.tanh(self.linear(self.layer_norm(x)))
        elif norm == 2:
            return self.tanh(self.linear(self.batch_norm(x)))

    def _init_weight_bias_linear_increasing(self, in_seq_len):
        weight_init = torch.arange(1, self.linear.weight.numel() + 1).float()
        weight_init /= in_seq_len
        weight_init = weight_init.view(self.linear.out_features,
                                       self.linear.in_features)
        self.linear.weight = nn.Parameter(weight_init)
        bias_init = torch.zeros(self.linear.bias.size())
        self.linear.bias = nn.Parameter(bias_init)


class BinarizeParams(torch.autograd.Function):
    """
    Binarize the weights, biases, and activations of a neural network
    """
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # saturated Straight-Through Estimator (STE)
        grad_input = F.hardtanh(grad_output)
        return grad_input


class BinarizedLinear(nn.Module):
    """
    Single Binarized Linear Layer

    Example usage:
        batch_size = 4
        in_seq_len = 10
        out_seq_len = 3
        binarized_linear = BinarizedLinear(in_seq_len, out_seq_len)
        input_b = torch.randint(0, 2, (batch_size, in_seq_len)) * 2.0 - 1
        output_b = binarized_linear(input_b)  # input_b entries are 1.0 or -1.0
    """

    def __init__(self, in_seq_len, out_seq_len):
        super().__init__()
        self.weight = nn.Parameter(
                torch.randint(0, 2, (out_seq_len, in_seq_len)) * 2.0 - 1)
        # Don't add bias!

    def forward(self, input_b):
        """
        input_b and output_b entries are 1.0 or -1.0 (dtype is Float)

        shape of input_b : (batch_size, in_seq_len)
        shape of weight_b: (out_seq_len, in_seq_len)
        shape of output_b: (batch_size, out_seq_len)
        """
        weight_b = BinarizeParams.apply(self.weight)
        linear_output = F.linear(input_b, weight_b)  # Don't add bias!
        output_b = BinarizeParams.apply(linear_output)
        return output_b


class Naive(nn.Module):
    """
    Forecast 1 if moving average of last 60 minutes log return is positive,
    else -1. (Follow last hour trend)
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.hardtanh = nn.Hardtanh(min_val=-0.0001, max_val=0.0001)

    def forward(self, x):
        """
        shape of input : (batch_size, in_seq_len) where in_seq_len >= 60
        shape of output: (batch_size, 1)
        """
        assert x.shape[1] >= 60, "in_seq_len must be at least 60"
        ma = x[:, -60:].mean(dim=1, keepdim=True)
        return self.hardtanh(self.linear(ma)) * 10000
