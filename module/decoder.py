import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self,latent_dim:int): 
        super().__init__()
        self.linear=nn.Linear(in_features=2*latent_dim,out_features=1)
    def forward(self,
            z:torch.Tensor,
            h:torch.Tensor
        ):
        x=torch.cat([z,h],dim=-1) 
        y=self.linear(x)
        return y

class Edge_Wise_Scoring_Network(nn.Module):
    def __init__(self,latent_dim:int,edge_dim:int): 
        super().__init__()
        self.linear_1=nn.Linear(in_features=2*latent_dim+edge_dim,out_features=latent_dim)
        self.linear_2=nn.Linear(in_features=latent_dim,out_features=1)
        self.relu=nn.ReLU()
    def forward(self,
            h:torch.Tensor,
            edge_attr:torch.Tensor,
            edge_index:torch.Tensor
        ): 
        src,tar=edge_index
        h_tar=h[tar]
        h_src=h[src]
        p_m=torch.cat([h_tar,h_src,edge_attr],dim=-1)
        p_m=self.linear_1(p_m)
        p_m=self.relu(p_m)
        edge_score=self.linear_2(p_m)
        return edge_score # [E,1]
