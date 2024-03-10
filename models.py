import torch
import torch.nn as nn


class OneLinear(nn.Module):
    def __init__(self, in_seq_len, out_seq_len):
        """
        in_seq_len: input sequence length
        out_seq_len: output sequence length
        """
        super().__init__()
        self.linear = nn.Linear(in_seq_len, out_seq_len)
        # comment out next line to use default initialization
        self._init_weight_bias_linear_increasing(in_seq_len)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        dim of input : (in_seq_len)
        dim of output: (out_seq_len)
        """
        return self.tanh(self.linear(x))

    def _init_weight_bias_linear_increasing(self, in_seq_len):
        weight_init = torch.arange(1, self.linear.weight.numel() + 1).float()
        weight_init /= in_seq_len
        weight_init = weight_init.view(self.linear.out_features,
                                       self.linear.in_features)
        self.linear.weight = nn.Parameter(weight_init)
        bias_init = torch.zeros(self.linear.bias.size())
        self.linear.bias = nn.Parameter(bias_init)
