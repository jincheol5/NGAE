import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,
            input_dim:int,
            latent_dim:int,
            encode_dim:int
        ): 
        super().__init__()
        self.linear=nn.Linear(in_features=input_dim+latent_dim,out_features=encode_dim)
    def forward(self,
            x:torch.Tensor,
            h:torch.Tensor
        ):
        x=torch.cat([x,h],dim=-1)
        y=self.linear(x)
        return y 