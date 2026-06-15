import torch
from torch import nn

class CADArgsSideEmbedding(nn.Module) :
    """
    Encodes arguments sequence to a d_model dimensional representation using a Transformer Encoder.
    """
    def __init__(self,
                n_args: int,
                d_model: int = 512,
                max_len: int = 1024):
    
        super().__init__()
        self.arg_embedding = nn.Linear(n_args, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        
    def forward(self, arg_sequences) :
        """
        Args:
            arg_sequences: (B, T_args, n_args)
        """
        seq_len = arg_sequences.size(1)
        positions = torch.arange(seq_len, device=arg_sequences.device).unsqueeze(0)
        arg_embeds = self.arg_embedding(arg_sequences)
        pos_embeds = self.pos_embedding(positions)
        return arg_embeds + pos_embeds