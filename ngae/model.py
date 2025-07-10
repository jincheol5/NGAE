import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from .model_train_utils import ModelTrainUtils

"""
Encoder
"""
class Encoder(torch.nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.linear=nn.Linear(in_features=node_dim+latent_dim,out_features=latent_dim)
    def forward(self,x,h):
        z=self.linear(torch.cat([x,h],dim=-1))
        return z

"""
Processor
1. MPNN
2. GAT
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

"""
Decoder
1. Decoder
2. Predecessor
3. Terminator
"""
class Decoder(torch.nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.linear=nn.Linear(in_features=2*latent_dim,out_features=1)
    def forward(self,z,h):
        y=self.linear(torch.cat([z,h],dim=-1))
        return y # [N,1]

class Predecessor(torch.nn.Module):
    def __init__(self,latent_dim,edge_dim):
        super().__init__()
        self.edge_wise_scoring=nn.Sequential(
            nn.Linear(2*latent_dim+edge_dim,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,1) 
        )
    def forward(self,z,h,edge_index,edge_attr):
        x=torch.cat([z,h],dim=-1)
        src,dst=edge_index
        x_j=x[src]
        x_i=x[dst]
        score=self.edge_wise_scoring(torch.cat([x_i,x_j,edge_attr],dim=-1))
        return score # [E,1]

class Terminator(torch.nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.linear=nn.Linear(in_features=latent_dim+latent_dim,out_features=1)
    def forward(self,h):
        h_mean=h.mean(dim=0) # [latent_dim,]
        h_mean=h_mean.unsqueeze(0).expand(h.size(0),-1) # [N,latent_dim]
        h_concat=torch.cat([h,h_mean],dim=-1)
        tau=self.linear(h_concat) # [N,1]
        return tau.mean(dim=0,keepdim=True) # [1,1]


"""
NGAE
"""
class NGAE_BF(torch.nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim):
        super().__init__()
        self.encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.processor=MPNN_Processor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.decoder=Decoder(latent_dim=latent_dim)
        self.predecessor=Predecessor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.terminator=Terminator(latent_dim=latent_dim)

    def forward(self,trajectory,h_0,edge_index,edge_attr,task="train"):
        seq_len,num_nodes,node_dim=trajectory.size()



        pre_h=h_0
        x=trajectory[0]
        for i in range(seq_len-1): # To do 3
            z=self.encoder(x=x,h=pre_h)
            h=self.processor(x=z,edge_index=edge_index,edge_attr=edge_attr)
            y=self.decoder(x=z,h=h)
            edge_score=self.predecessor(z=z,h=h,edge_index=edge_index,edge_attr=edge_attr)
            tau=self.terminator(h=h)

            """
            set next x, pre_h
            """
            x=ModelTrainUtils.teacher_forcing(pred=y,label=trajectory[i+1],p=0.5)
            pre_h=h
"""
To do.
1. edge_score -> [N,1] p_idx 변환 함수 
2. [seq_len,N,1] 형태 tensor로부터 loss 계산
3. input으로 seq_len-1이 들어가야 할지 seq_len이 들어가야 할지?
"""