import torch
import torch.nn as nn
from typing_extensions import Literal
from torch_geometric.nn import MessagePassing,GATConv,GATv2Conv
from torch_geometric.utils import subgraph
from .model_train_utils import ModelTrainUtils

"""
Encoder, Decoder
"""
class Encoder(nn.Module):
    def __init__(self,input_dim,latent_dim): 
        super().__init__()
        self.linear=nn.Linear(in_features=input_dim+latent_dim,out_features=latent_dim)
    def forward(self,x,h):
        x=torch.cat([x,h],dim=-1)
        y=self.linear(x)
        return y 

class Decoder(nn.Module):
    def __init__(self,latent_dim): 
        super().__init__()
        self.linear=nn.Linear(in_features=2*latent_dim,out_features=1)
    def forward(self,z,h):
        x=torch.cat([z,h],dim=-1) 
        y=self.linear()
        return y

class Predecessor_Decoder(nn.Module):
    def __init__(self,latent_dim,edge_dim): 
        super().__init__()
        self.linear_1=nn.Linear(in_features=2*latent_dim+edge_dim,out_features=latent_dim)
        self.linear_2=nn.Linear(in_features=latent_dim,out_features=1)
        self.relu=nn.ReLU()
    def forward(self,h,edge_attr,edge_index): 
        src,tar=edge_index
        h_tar=h[tar]
        h_src=h[src]
        p_m=torch.cat([h_tar,h_src,edge_attr],dim=-1)
        p_m=self.linear_1(p_m)
        p_m=self.relu(p_m)
        edge_score=self.linear_2(p_m)
        return edge_score # [E,1]

class Terminator(torch.nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.linear=nn.Linear(in_features=latent_dim+latent_dim,out_features=1)
    def forward(self,h,num_nodes): # h=[sub_B*N,latent_dim]
        sub_batch_size=h.size(0)//num_nodes
        h_reshaped=h.view(sub_batch_size,num_nodes,-1) # [sub_B,N,latent_dim]
        h_mean_g=h_reshaped.mean(dim=1) # [sub_B,latent_dim]
        h_mean_per_node=h_mean_g.unsqueeze(1).expand(-1,num_nodes,-1) # [sub_B,N,latent_dim]
        h_concat=torch.cat([h_reshaped,h_mean_per_node],dim=-1) # [sub_B,N,2*latent_dim]
        tau_per_node=self.linear(h_concat) # [sub_B,N,1]
        tau=tau_per_node.mean(dim=1).squeeze(-1) # [sub_B,]
        return tau # [sub_B,]

"""
Processor
1. MPNN (max,sum,mean)
2. GAT
3. GATv2
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
        aggr_out=torch.nan_to_num(aggr_out,neginf=0.0,posinf=0.0)
        readout=self.readout_linear(torch.cat([x,aggr_out],dim=-1))
        return readout
    def forward(self,x,edge_index,edge_attr):
        h=self.propagate(edge_index=edge_index,x=x,edge_attr=edge_attr)
        return h

class GAT_Processor(nn.Module):
    def __init__(self,latent_dim,edge_dim):
        super().__init__()
        self.gat=GATConv(in_channels=latent_dim,out_channels=latent_dim,edge_dim=edge_dim,add_self_loops=False)
        self.relu=nn.ReLU()
    def forward(self,x,edge_index,edge_attr):
        h=self.gat(x,edge_index,edge_attr)
        return self.relu(h)

class GATv2_Processor(nn.Module):
    def __init__(self,latent_dim,edge_dim):
        super().__init__()
        self.gat=GATv2Conv(in_channels=latent_dim,out_channels=latent_dim,edge_dim=edge_dim,add_self_loops=False)
        self.relu=nn.ReLU()
    def forward(self,x,edge_index,edge_attr):
        h=self.gat(x,edge_index,edge_attr)
        return self.relu(h)

"""
NGAE Module
"""
class NGAE(nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim,processor:Literal['mpnn','gat','gatv2']='mpnn',aggr:Literal['max','sum','mean']='max'):
        super().__init__()
        self.algo_encoder=Encoder(input_dim=node_dim,latent_dim=latent_dim)
        match processor:
            case 'mpnn':
                self.processor=MPNN_Processor(latent_dim=latent_dim,edge_dim=edge_dim,aggr=aggr)
            case 'gat':
                self.processor=GAT_Processor(latent_dim=latent_dim,edge_dim=edge_dim)
            case 'gatv2':
                self.processor=GATv2_Processor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.algo_decoder=Decoder(latent_dim=latent_dim)
        self.p_decoder=Predecessor_Decoder(latent_dim=latent_dim,edge_dim=edge_dim)
        self.terminator=Terminator(latent_dim=latent_dim)
    
    def forward(self,algo_trajectory,algo_mask_trajectory,h_0,edge_index,edge_attr,batch,algo_type:Literal['bfs','bf']='bfs',mode:Literal['train','test']='train'):
        """
        algo_trajectory=[max_seq_len,B*N,1]
        algo_mask_trajectory=[max_seq_len,B*N,1]
        pos=[B*N,1]
        h_0=[B*N,latent_dim]
        batch=[B*N,]
        """
        pred_algo_logit_list=[]
        pred_edge_score_logit_list=[]
        pred_tau_logit_list=[]
        sub_edge_index_list=[]

        seq_len=algo_trajectory.size(0)
        batch_size=batch.unique().numel()
        num_nodes=algo_trajectory.size(1)//batch_size
        pre_h=h_0 # [B*N,latent_dim]
        algo=algo_trajectory[0] # [B*N,1]
        algo_mask=algo_mask_trajectory[0].squeeze(-1) # [B*N,]

        for i in range(seq_len-1):
            # compute subgraph
            sub_edge_index,sub_edge_attr=subgraph(subset=algo_mask,edge_index=edge_index,edge_attr=edge_attr,relabel_nodes=True)
            sub_edge_index_list.append(sub_edge_index)
            sub_algo=algo[algo_mask] # [sub_B*N,1]
            sub_pre_h=pre_h[algo_mask] # [sub_B*N,latent_dim]

            # encode
            encoded_algo=self.algo_encoder(x=sub_algo,h=sub_pre_h) # [sub_B*N,latent_dim]

            # process
            h=self.processor(x=encoded_algo,edge_index=sub_edge_index,edge_attr=sub_edge_attr) # [sub_B*N,latent_dim]

            # decode
            decoded_algo_logit=self.algo_decoder(z=encoded_algo,h=h) # [sub_B*N,1]
            decoded_edge_score_logit=self.p_decoder(h=h,edge_attr=sub_edge_attr,edge_index=sub_edge_index) # [sub_E,1]
            decoded_tau_logit=self.terminator(h=h,num_nodes=num_nodes) # [sub_B,]

            # stack output
            pred_algo_logit_list.append(decoded_algo_logit)
            pred_edge_score_logit_list.append(decoded_edge_score_logit)
            pred_tau_logit_list.append(decoded_tau_logit)

            # set next algo,p,pre_h
            match mode:
                case 'train':
                    ### algo
                    if algo_type=='bfs':
                        pred_algo=ModelTrainUtils.compute_r_from_logit(logit=decoded_algo_logit) # [sub_B*N,1]
                    else: # bf
                        pred_algo=decoded_algo_logit # [sub_B*N,1]
                    restored_pred_algo=ModelTrainUtils.restore_output(output=pred_algo,mask=algo_mask) # [B*N,1]
                    algo=ModelTrainUtils.teacher_forcing(pred=restored_pred_algo,label=algo_trajectory[i+1],p=0.5) # [B*N,1]
                case 'test':
                    ### algo
                    if algo_type=='bfs':
                        pred_algo=ModelTrainUtils.compute_r_from_logit(logit=decoded_algo_logit) # [sub_B*N,1]
                    else: # bf
                        pred_algo=decoded_algo_logit # [sub_B*N,1]
                    algo=ModelTrainUtils.restore_output(output=pred_algo,mask=algo_mask) # [B*N,1]

            next_h=ModelTrainUtils.restore_output(output=h,mask=algo_mask,feature_dim=pre_h.size(-1)) # [B*N,latent_dim]
            pre_h=next_h
            algo_mask=algo_mask_trajectory[i+1].squeeze(-1) # [B*N,]

        # return output
        output={}
        output['algo']=pred_algo_logit_list # List of [sub_B*N,1]
        output['edge_score']=pred_edge_score_logit_list # List of [sub_E,1]
        output['tau']=pred_tau_logit_list # List of [sub_B,]
        output['sub_edge_index']=sub_edge_index_list # List of sub_edge_index
        return output