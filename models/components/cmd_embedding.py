import torch
from torch import nn


class CADCmdSideEmbedding(nn.Module) :
    """
    Encodes command sequence to a d_model dimensional representation using a Transformer Encoder.
    """
    def __init__(self,
                vocab_size: int,
                d_model: int = 512,
                max_len: int = 1024):
    
        super().__init__()
        self.cmd_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        
    def forward(self, cmd_input_ids) :
        seq_len = cmd_input_ids.size(1)
        positions = torch.arange(seq_len, device=cmd_input_ids.device).unsqueeze(0)
        cmd_embeds = self.cmd_embedding(cmd_input_ids)
        pos_embeds = self.pos_embedding(positions)
        return cmd_embeds + pos_embeds
