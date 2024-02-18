import torch.nn as nn


class TwoLinear(nn.Module):
    def __init__(self, n_features, in_seq_len, out_seq_len):
        super().__init__()
        self.linear1 = nn.Linear(in_seq_len, out_seq_len)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_features, 1)

    def forward(self, x):
        """
        dim of input : (batch_size, n_features, in_seq_len)
        dim of output: (batch_size, out_seq_len)"""
        hidden = self.relu(self.linear1(x))
        output = self.linear2(hidden)
        return output
