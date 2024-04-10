import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    import torch
    def __init__(self, in_seq_len, out_seq_len, batch_size, nhead=1, num_encoder_layers=3, num_decoder_layers=3, d_model=1, device ='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first=True).to(self.device)
        
        self.tgt = torch.zeros(batch_size, out_seq_len, d_model, device = self.device)

    

    def forward(self, src):
        # Add dimension to the original form of data
        src = src.to(self.device).unsqueeze(-1)

        # Process through the transformer
        output = self.transformer(src, self.tgt)

        # Cut the dimension back to the original form of y
        output = output.squeeze(-1) 

        return output
