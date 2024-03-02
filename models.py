import torch.nn as nn


class OneLinear(nn.Module):
    def __init__(self, in_seq_len, out_seq_len):
        """
        in_seq_len: input sequence length
        out_seq_len: output sequence length
        """
        super().__init__()
        self.linear = nn.Linear(in_seq_len, out_seq_len)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        dim of input : (in_seq_len)
        dim of output: (out_seq_len)
        """
        return self.tanh(self.linear(x))
