import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal
from typing import Optional,Tuple,Union
from torch import Tensor
from torch_geometric.nn import MessagePassing,GATConv,GATv2Conv
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter_max
from .model_train_utils import ModelTrainUtils


"""
Encoder, Decoder
"""
class Encoder(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.linear=nn.Linear(in_features=node_dim+latent_dim,out_features=latent_dim)
    def forward(self,x,h):
        z=self.linear(torch.cat([x,h],dim=-1))
        return z # [N,latend_dim]

class Algo_Decoder(nn.Module):
    def __init__(self,latent_dim): 
        super().__init__()
        self.linear=nn.Linear(in_features=2*latent_dim,out_features=1)
    def forward(self,z,h):
        y=self.linear(torch.cat([z,h],dim=-1))
        return y # [N,1]

class Predecessor_Decoder(torch.nn.Module):
    def __init__(self,latent_dim,edge_dim):
        super().__init__()
        self.edge_wise_scoring=nn.Sequential(
            nn.Linear(2*latent_dim+edge_dim,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,1) 
        )
    def forward(self,h,edge_index,edge_attr):
        src,dst=edge_index
        h_j=h[src]
        h_i=h[dst]
        score=self.edge_wise_scoring(torch.cat([h_i,h_j,edge_attr],dim=-1))
        return score # [E,1]

class Terminator(torch.nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.linear=nn.Linear(in_features=2*latent_dim,out_features=1)
    def forward(self,h):
        h_mean=h.mean(dim=0) # [latent_dim,]
        h_mean=h_mean.unsqueeze(0).expand(h.size(0),-1) # [N,latent_dim]
        h_concat=torch.cat([h,h_mean],dim=-1)
        tau=self.linear(h_concat) # [N,1]
        return tau.mean(dim=0,keepdim=True).view(-1) # [1,]

"""
Processor
1. MPNN
2. GAT
"""
class MPNN_Processor(MessagePassing):
    def __init__(self,latent_dim,edge_dim,aggr='max'):
        super().__init__(aggr=aggr)
        self.message_linear=nn.Linear(in_features=2*latent_dim+edge_dim,out_features=latent_dim)
        self.readout_linear=nn.Linear(in_features=2*latent_dim,out_features=latent_dim)
    
    def message(self,x_i,x_j,edge_attr):
        message=self.message_linear(torch.cat([x_i,x_j,edge_attr],dim=-1))
        return message
    
    def update(self,aggr_out,x):
        readout=self.readout_linear(torch.cat([x,aggr_out],dim=-1))
        return readout

    def forward(self,x,edge_index,edge_attr):
        h=self.propagate(edge_index=edge_index,x=x,edge_attr=edge_attr)
        return h

"""
NGAE Module
"""
class NGAE(nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim,algorithm:Literal['bfs','bf']='bfs',processor:Literal['mpnn','gat','gatv2']='mpnn',aggr:Literal['max','sum','mean']='max'):
        super().__init__()
        self.algorithm=algorithm
        self.algo_encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        match processor:
            case 'mpnn':
                self.processor=MPNN_Processor(latent_dim=latent_dim,edge_dim=latent_dim,aggr=aggr)
            case 'gat':
                self.processor=GATConv(in_channels=latent_dim,out_channels=latent_dim,edge_dim=edge_dim)
            case 'gatv2':
                self.processor=GATv2Conv(in_channels=latent_dim,out_channels=latent_dim,edge_dim=edge_dim)
        self.algo_decoder=Algo_Decoder(latent_dim=latent_dim)
        self.p_decoder=Predecessor_Decoder(latent_dim=latent_dim,edge_dim=edge_dim)
        self.terminator=Terminator(latent_dim=latent_dim)
    
    def forward(self,algo_trajectory,h_0,edge_index,edge_attr,mode:Literal['train','test']='train'):
        pred_algo_list=[]
        pred_edge_score_list=[]
        pred_tau_list=[]

        seq_len,_,_=algo_trajectory.size()
        pre_h=h_0
        algo=algo_trajectory[0]
        for i in range(seq_len-1):
            # encode
            encoded_algo=self.algo_encoder(x=algo,h=pre_h) # [N,latent_dim]

            # process
            h=self.processor(x=encoded_algo,edge_index=edge_index,edge_attr=edge_attr) # [N,latent_dim]

            # decode
            decoded_algo_logit=self.algo_decoder(z=encoded_algo,h=h) # [N,1] logits
            decoded_tau_logit=self.terminator(h=h) # [1,]
            if self.algorithm=='bf':
                decoded_edge_score_logit=self.p_decoder(h=h,edge_index=edge_index,edge_attr=edge_attr) # [E,1] edge scores
            

            # stack output
            pred_algo_list.append(decoded_algo_logit)
            pred_tau_list.append(decoded_tau_logit)
            if self.algorithm=='bf':
                pred_edge_score_list.append(decoded_edge_score_logit)

            # set next algo,pre_h
            match mode:
                case 'train':
                    algo=ModelTrainUtils.teacher_forcing(pred=decoded_algo_logit,label=algo_trajectory[i+1],p=0.5)
                case 'test':
                    if self.algorithm=='bfs':
                        algo=ModelTrainUtils.compute_r_from_logit(logit=decoded_algo_logit)
                    else: # bf
                        algo=decoded_algo_logit
            pre_h=h
        """
        return output
            -all output is logit
        """
        output={}
        output['algo']=torch.stack(pred_algo_list,dim=0) # [seq_len-1,N,1]
        output['tau']=torch.stack(pred_tau_list,dim=0) # [seq_len-1,1]
        if self.algorithm=='bf':
            output['edge_score']=torch.stack(pred_edge_score_list,dim=0) # [seq_len-1,E,1]
        return output