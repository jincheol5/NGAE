import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,Tuple,Union
from torch import Tensor
from torch_geometric.nn import MessagePassing,GATConv
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter_max
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
        return F.relu(m)
    
    def update(self,aggr_out,x):
        u=self.U(torch.cat([x,aggr_out],dim=-1))
        return F.relu(u)

    def forward(self,x,edge_index,edge_attr):
        h=self.propagate(edge_index=edge_index,x=x,edge_attr=edge_attr)
        return h

class GAT_GumbelSoftmax(GATConv):
    """
    PyG GATConv 상속하여 gumbel-softmax 적용을 위한 edge_update 오버라이딩
    """
    def __init__(
        self,
        in_channels: Union[int,Tuple[int,int]],
        out_channels: int,
        edge_dim: Optional[int]=None,
        tau: float=1.0,
        hard: bool=False,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            **kwargs,
        )
        self.tau=tau
        self.hard=hard

    def edge_update(self,alpha_j: Tensor,alpha_i: OptTensor,edge_attr: OptTensor,index: Tensor,ptr: OptTensor,dim_size: Optional[int],) -> Tensor:
        alpha=alpha_j if alpha_i is None else alpha_j+alpha_i
        if index.numel()==0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim()==1:
                edge_attr=edge_attr.view(-1,1)
            edge_feat=self.lin_edge(edge_attr)
            edge_feat=edge_feat.view(-1,self.heads,self.out_channels)
            alpha_edge=(edge_feat*self.att_edge).sum(dim=-1)
            alpha=alpha+alpha_edge

        alpha=F.leaky_relu(alpha,self.negative_slope)

        # Gumbel noise + 온도 적용 (학습 시에만 noise, 추론 시는 noise 없이)
        if self.training:
            uniform=torch.rand_like(alpha)
            gumbel=-torch.log(-torch.log(uniform+1e-9)+1e-9)
            logits=(alpha+gumbel)/self.tau
        else:
            logits=alpha/self.tau

        # 그룹별 soft attention (softmax)
        soft_attn=softmax(logits,index,ptr,num_nodes=dim_size)

        # hard 모드라면 straight-through one-hot 생성
        if self.hard:
            # 그룹별 최대값 계산
            max_per_idx,_=scatter_max(logits,index,dim=0,dim_size=dim_size)
            max_at_edges=max_per_idx[index]  # 각 edge에 대응하는 그룹별 max
            hard_attn=(logits==max_at_edges).type_as(soft_attn)

            if self.training:
                # straight-through trick
                attn=(hard_attn-soft_attn).detach()+soft_attn
            else:
                # 추론 시 pure hard
                attn=hard_attn
        else:
            # soft 모드
            attn=soft_attn
        attn=F.dropout(attn,p=self.dropout,training=self.training)
        return attn

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
    """
    NGAE paper 방식
    """
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

# class Predecessor(torch.nn.Module):
#     """
#     clrs 방식
#     """
#     def __init__(self,latent_dim,edge_dim):
#         super().__init__()
#         self.p1=nn.Linear(latent_dim,latent_dim)
#         self.p2=nn.Linear(latent_dim,latent_dim)
#         self.p3=nn.Linear(edge_dim,latent_dim)
#         self.p4=nn.Linear(latent_dim,1)
#     def forward(self,h,edge_index,edge_attr):
#         src,dst=edge_index
#         p1_all=self.p1(h)
#         p2_all=self.p2(h)
#         p1_dst=p1_all[dst] 
#         p2_src=p2_all[src]
#         p3=self.p3(edge_attr)
#         p_e=p2_src+p3
#         p_m=torch.max(p1_dst,p_e)         
#         score=self.p4(p_m) 
#         return score # [E,1]

class Terminator(torch.nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.linear=nn.Linear(in_features=latent_dim+latent_dim,out_features=1)
    def forward(self,h):
        h_mean=h.mean(dim=0) # [latent_dim,]
        h_mean=h_mean.unsqueeze(0).expand(h.size(0),-1) # [N,latent_dim]
        h_concat=torch.cat([h,h_mean],dim=-1)
        tau=self.linear(h_concat) # [N,1]
        return tau.mean(dim=0,keepdim=True).view(-1) # [1,]


"""
NGAE
1. NGAE_MPNN_BFS
2. NGAE_MPNN_BF
3. NGAE_MPNN
4. NGAE_GAT_BFS
5. NGAE_GAT_BF
6. NGAE_GAT
"""
class NGAE_MPNN_BFS(torch.nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim):
        super().__init__()
        self.encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.processor=MPNN_Processor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.decoder=Decoder(latent_dim=latent_dim)
        self.terminator=Terminator(latent_dim=latent_dim)

    def forward(self,algo_trajectory,h_0,edge_index,edge_attr,mode="train"):
        pred_y_list=[]
        pred_tau_list=[]

        seq_len,_,_=algo_trajectory.size()
        pre_h=h_0
        x=algo_trajectory[0]
        for i in range(seq_len-1):
            z=self.encoder(x=x,h=pre_h)
            h=self.processor(x=z,edge_index=edge_index,edge_attr=edge_attr)
            y=self.decoder(z=z,h=h)
            tau=self.terminator(h=h)

            """
            stack output
            """
            pred_y_list.append(y)
            pred_tau_list.append(tau)

            """
            set next x, pre_h
            """
            pred_y=ModelTrainUtils.compute_BFS_from_logit(logit=y)
            match mode:
                case 'train':
                    x=ModelTrainUtils.teacher_forcing(pred=pred_y,label=algo_trajectory[i+1],p=0.5)
                case 'test':
                    x=pred_y
            pre_h=h
        """
        return output
            -all output is logit
        """
        output={}
        output['y']=torch.stack(pred_y_list,dim=0) # [seq_len-1,N,1]
        output['tau']=torch.stack(pred_tau_list,dim=0) # [seq_len-1,1]
        return output

class NGAE_MPNN_BF(torch.nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim):
        super().__init__()
        self.encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.processor=MPNN_Processor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.decoder=Decoder(latent_dim=latent_dim)
        self.predecessor=Predecessor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.terminator=Terminator(latent_dim=latent_dim)

    def forward(self,algo_trajectory,h_0,edge_index,edge_attr,mode="train"):
        pred_y_list=[]
        pred_edge_score_list=[]
        pred_tau_list=[]

        seq_len,_,_=algo_trajectory.size()
        pre_h=h_0
        x=algo_trajectory[0]
        for i in range(seq_len-1):
            z=self.encoder(x=x,h=pre_h)
            h=self.processor(x=z,edge_index=edge_index,edge_attr=edge_attr)
            y=self.decoder(z=z,h=h)
            edge_score=self.predecessor(h=h,edge_index=edge_index,edge_attr=edge_attr)
            tau=self.terminator(h=h)

            """
            stack output
            """
            pred_y_list.append(y)
            pred_edge_score_list.append(edge_score)
            pred_tau_list.append(tau)

            """
            set next x, pre_h
            """
            match mode:
                case 'train':
                    x=ModelTrainUtils.teacher_forcing(pred=y,label=algo_trajectory[i+1],p=0.5)
                case 'test':
                    x=y
            pre_h=h
        """
        return output
            -all output is logit
        """
        output={}
        output['y']=torch.stack(pred_y_list,dim=0) # [seq_len-1,N,1]
        output['edge_score']=torch.stack(pred_edge_score_list,dim=0) # [seq_len-1,E,1]
        output['tau']=torch.stack(pred_tau_list,dim=0) # [seq_len-1,1]
        return output

class NGAE_MPNN(torch.nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim):
        super().__init__()
        self.bfs_encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.bf_encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.processor=MPNN_Processor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.bfs_decoder=Decoder(latent_dim=latent_dim)
        self.bf_decoder=Decoder(latent_dim=latent_dim)
        self.predecessor=Predecessor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.bfs_terminator=Terminator(latent_dim=latent_dim)
        self.bf_terminator=Terminator(latent_dim=latent_dim)
    
    def forward(self,bfs_trajectory,bf_trajectory,h_0,edge_index,edge_attr,mode="train"):
        pred_bfs_list=[]
        pred_bf_list=[]
        pred_edge_score_list=[]
        pred_bfs_tau_list=[]
        pred_bf_tau_list=[]

        bfs_seq_len,_,_=bfs_trajectory.size()
        bf_seq_len,_,_=bf_trajectory.size()
        seq_len=max(bfs_seq_len,bf_seq_len)
        pre_bfs_h=h_0
        pre_bf_h=h_0
        bfs_x=bfs_trajectory[0]
        bf_x=bf_trajectory[0]
        for i in range(seq_len-1):
            if i<=bfs_seq_len-2:
                bfs_z=self.bfs_encoder(x=bfs_x,h=pre_bfs_h)
                bfs_h=self.processor(x=bfs_z,edge_index=edge_index,edge_attr=edge_attr)
                bfs_y=self.bfs_decoder(z=bfs_z,h=bfs_h)
                bfs_tau=self.bfs_terminator(h=bfs_h)

                # stack output
                pred_bfs_list.append(bfs_y)
                pred_bfs_tau_list.append(bfs_tau)

                # set next x, pre_h
                pred_y=ModelTrainUtils.compute_BFS_from_logit(logit=bfs_y)
                match mode:
                    case 'train':
                        bfs_x=ModelTrainUtils.teacher_forcing(pred=pred_y,label=bfs_trajectory[i+1],p=0.5)
                    case 'test':
                        bfs_x=pred_y
                pre_bfs_h=bfs_h

            if i<=bf_seq_len-2:
                bf_z=self.bf_encoder(x=bf_x,h=pre_bf_h)
                bf_h=self.processor(x=bf_z,edge_index=edge_index,edge_attr=edge_attr)
                bf_y=self.bf_decoder(z=bf_z,h=bf_h)
                edge_score=self.predecessor(h=bf_h,edge_index=edge_index,edge_attr=edge_attr)
                bf_tau=self.bf_terminator(h=bf_h)

                # stack output
                pred_bf_list.append(bf_y)
                pred_edge_score_list.append(edge_score)
                pred_bf_tau_list.append(bf_tau)

                # set next x, pre_h
                match mode:
                    case 'train':
                        x=ModelTrainUtils.teacher_forcing(pred=bf_y,label=bf_trajectory[i+1],p=0.5)
                    case 'test':
                        x=bf_y
                pre_bf_h=bf_h
        """
        return output
            -all output is logit
        """
        output={}
        output['bfs_y']=torch.stack(pred_bfs_list,dim=0) # [bfs_seq_len-1,N,1]
        output['bfs_tau']=torch.stack(pred_bfs_tau_list,dim=0) # [bfs_seq_len-1,1]

        output['bf_y']=torch.stack(pred_bf_list,dim=0) # [bf_seq_len-1,N,1]
        output['edge_score']=torch.stack(pred_edge_score_list,dim=0) # [bf_seq_len-1,E,1]
        output['bf_tau']=torch.stack(pred_bf_tau_list,dim=0) # [bf_seq_len-1,1]
        return output

class NGAE_GAT_BFS(torch.nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim):
        super().__init__()
        self.encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.processor=GAT_GumbelSoftmax(in_channels=latent_dim,out_channels=latent_dim,edge_dim=edge_dim,tau=0.5)
        self.decoder=Decoder(latent_dim=latent_dim)
        self.terminator=Terminator(latent_dim=latent_dim)

    def forward(self,algo_trajectory,h_0,edge_index,edge_attr,mode="train"):
        pred_y_list=[]
        pred_tau_list=[]

        seq_len,_,_=algo_trajectory.size()
        pre_h=h_0
        x=algo_trajectory[0]
        for i in range(seq_len-1):
            z=self.encoder(x=x,h=pre_h)
            h=self.processor(x=z,edge_index=edge_index,edge_attr=edge_attr)
            y=self.decoder(z=z,h=h)
            tau=self.terminator(h=h)

            """
            stack output
            """
            pred_y_list.append(y)
            pred_tau_list.append(tau)

            """
            set next x, pre_h
            """
            pred_y=ModelTrainUtils.compute_BFS_from_logit(logit=y)
            match mode:
                case 'train':
                    x=ModelTrainUtils.teacher_forcing(pred=pred_y,label=algo_trajectory[i+1],p=0.5)
                case 'test':
                    x=pred_y
            pre_h=h
        """
        return output
            -all output is logit
        """
        output={}
        output['y']=torch.stack(pred_y_list,dim=0) # [seq_len-1,N,1]
        output['tau']=torch.stack(pred_tau_list,dim=0) # [seq_len-1,1]
        return output

class NGAE_GAT_BF(torch.nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim):
        super().__init__()
        self.encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.processor=GAT_GumbelSoftmax(in_channels=latent_dim,out_channels=latent_dim,edge_dim=edge_dim,tau=0.5)
        self.decoder=Decoder(latent_dim=latent_dim)
        self.predecessor=Predecessor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.terminator=Terminator(latent_dim=latent_dim)

    def forward(self,algo_trajectory,h_0,edge_index,edge_attr,mode="train"):
        pred_y_list=[]
        pred_edge_score_list=[]
        pred_tau_list=[]

        seq_len,_,_=algo_trajectory.size()
        pre_h=h_0
        x=algo_trajectory[0]
        for i in range(seq_len-1):
            z=self.encoder(x=x,h=pre_h)
            h=self.processor(x=z,edge_index=edge_index,edge_attr=edge_attr)
            y=self.decoder(z=z,h=h)
            edge_score=self.predecessor(h=h,edge_index=edge_index,edge_attr=edge_attr)
            tau=self.terminator(h=h)

            """
            stack output
            """
            pred_y_list.append(y)
            pred_edge_score_list.append(edge_score)
            pred_tau_list.append(tau)

            """
            set next x, pre_h
            """
            match mode:
                case 'train':
                    x=ModelTrainUtils.teacher_forcing(pred=y,label=algo_trajectory[i+1],p=0.5)
                case 'test':
                    x=y
            pre_h=h
        """
        return output
            -all output is logit
        """
        output={}
        output['y']=torch.stack(pred_y_list,dim=0) # [seq_len-1,N,1]
        output['edge_score']=torch.stack(pred_edge_score_list,dim=0) # [seq_len-1,E,1]
        output['tau']=torch.stack(pred_tau_list,dim=0) # [seq_len-1,1]
        return output

class NGAE_GAT(torch.nn.Module):
    def __init__(self,node_dim,edge_dim,latent_dim):
        super().__init__()
        self.bfs_encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.bf_encoder=Encoder(node_dim=node_dim,latent_dim=latent_dim)
        self.processor=GAT_GumbelSoftmax(in_channels=latent_dim,out_channels=latent_dim,edge_dim=edge_dim,tau=0.5)
        self.bfs_decoder=Decoder(latent_dim=latent_dim)
        self.bf_decoder=Decoder(latent_dim=latent_dim)
        self.predecessor=Predecessor(latent_dim=latent_dim,edge_dim=edge_dim)
        self.bfs_terminator=Terminator(latent_dim=latent_dim)
        self.bf_terminator=Terminator(latent_dim=latent_dim)
    
    def forward(self,bfs_trajectory,bf_trajectory,h_0,edge_index,edge_attr,mode="train"):
        pred_bfs_list=[]
        pred_bf_list=[]
        pred_edge_score_list=[]
        pred_bfs_tau_list=[]
        pred_bf_tau_list=[]

        bfs_seq_len,_,_=bfs_trajectory.size()
        bf_seq_len,_,_=bf_trajectory.size()
        seq_len=max(bfs_seq_len,bf_seq_len)
        pre_bfs_h=h_0
        pre_bf_h=h_0
        bfs_x=bfs_trajectory[0]
        bf_x=bf_trajectory[0]
        for i in range(seq_len-1):
            if i<=bfs_seq_len-2:
                bfs_z=self.bfs_encoder(x=bfs_x,h=pre_bfs_h)
                bfs_h=self.processor(x=bfs_z,edge_index=edge_index,edge_attr=edge_attr)
                bfs_y=self.bfs_decoder(z=bfs_z,h=bfs_h)
                bfs_tau=self.bfs_terminator(h=bfs_h)

                # stack output
                pred_bfs_list.append(bfs_y)
                pred_bfs_tau_list.append(bfs_tau)

                # set next x, pre_h
                pred_y=ModelTrainUtils.compute_BFS_from_logit(logit=bfs_y)
                match mode:
                    case 'train':
                        bfs_x=ModelTrainUtils.teacher_forcing(pred=pred_y,label=bfs_trajectory[i+1],p=0.5)
                    case 'test':
                        bfs_x=pred_y
                pre_bfs_h=bfs_h

            if i<=bf_seq_len-2:
                bf_z=self.bf_encoder(x=bf_x,h=pre_bf_h)
                bf_h=self.processor(x=bf_z,edge_index=edge_index,edge_attr=edge_attr)
                bf_y=self.bf_decoder(z=bf_z,h=bf_h)
                edge_score=self.predecessor(h=bf_h,edge_index=edge_index,edge_attr=edge_attr)
                bf_tau=self.bf_terminator(h=bf_h)

                # stack output
                pred_bf_list.append(bf_y)
                pred_edge_score_list.append(edge_score)
                pred_bf_tau_list.append(bf_tau)

                # set next x, pre_h
                match mode:
                    case 'train':
                        x=ModelTrainUtils.teacher_forcing(pred=bf_y,label=bf_trajectory[i+1],p=0.5)
                    case 'test':
                        x=bf_y
                pre_bf_h=bf_h
        """
        return output
            -all output is logit
        """
        output={}
        output['bfs_y']=torch.stack(pred_bfs_list,dim=0) # [bfs_seq_len-1,N,1]
        output['bfs_tau']=torch.stack(pred_bfs_tau_list,dim=0) # [bfs_seq_len-1,1]

        output['bf_y']=torch.stack(pred_bf_list,dim=0) # [bf_seq_len-1,N,1]
        output['edge_score']=torch.stack(pred_edge_score_list,dim=0) # [bf_seq_len-1,E,1]
        output['bf_tau']=torch.stack(pred_bf_tau_list,dim=0) # [bf_seq_len-1,1]
        return output