import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

"""
Processor
"""
class MPNN_Processor(MessagePassing):
    def __init__(self,latent_dim,edge_dim,aggr='max'):
        super().__init__(aggr=aggr)
        self.M=nn.Linear(in_features=latent_dim+latent_dim+edge_dim,out_features=latent_dim)
        self.U=nn.Linear(in_features=latent_dim+latent_dim,out_features=latent_dim)
    
    def message(self,x_i,x_j,edge_attr):
        m=self.M(torch.cat([x_i,x_j,edge_attr],dim=-1))
        return m
    
    def update(self,aggr_out,x):
        u=self.U(torch.cat([x,aggr_out],dim=-1))
        return u

    def forward(self,x,edge_index,edge_attr):
        h=self.propagate(edge_index=edge_index,x=x,edge_attr=edge_attr)
        return h