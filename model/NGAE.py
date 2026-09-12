import torch
import torch.nn as nn
from typing import Literal
from module import Encoder,MPNN_Processor,GAT_Processor,GATv2_Processor,Decoder,Edge_Wise_Scoring_Network

"""
여러 알고리즘을 학습하는 경우 single processor는 모든 알고리즘이 공유하고, 
encoder와 decoder는 각 알고리즘마다 별도로 존재
"""

class NGAE(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            encode_dim:int,
            latent_dim:int,
            processor:Literal["MPNN","GAT","GATv2"]="MPNN",
            aggr:Literal["max","add","mean"]="mean"
        ):
        super().__init__()
        self.encoder=nn.ModuleList([
            Encoder( # BFS Encoder
                input_dim=node_dim,
                latent_dim=latent_dim,
                encode_dim=encode_dim
            ),
            Encoder( # Bellman-Ford Encoder
                input_dim=node_dim,
                latent_dim=latent_dim,
                encode_dim=encode_dim
            )
        ])
        match processor:
            case "MPNN":
                self.processor=MPNN_Processor(
                    encode_dim=encode_dim,
                    latent_dim=latent_dim,
                    edge_dim=edge_dim,
                    aggr=aggr
                )
            case "GAT":
                self.processor=GAT_Processor(
                    encode_dim=encode_dim,
                    latent_dim=latent_dim,
                    edge_dim=edge_dim
                )
            case "GATv2":
                self.processor=GATv2_Processor(
                    encode_dim=encode_dim,
                    latent_dim=latent_dim,
                    edge_dim=edge_dim
                )
        self.decoder=nn.ModuleList([
            Decoder(latent_dim=latent_dim), # BFS Decoder
            Decoder(latent_dim=latent_dim) # Bellman-Ford Decoder
        ])
        self.edge_score_network=Edge_Wise_Scoring_Network(
            latent_dim=latent_dim,
            edge_dim=edge_dim
        )

    def forward(self,

        ):
        """
        """


class NGAE_BFS(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            encode_dim:int,
            latent_dim:int,
            processor:Literal["MPNN","GAT","GATv2"]="MPNN",
            aggr:Literal["max","add","mean"]="mean"
        ):
        super().__init__()
        self.encoder=Encoder(
            input_dim=node_dim,
            latent_dim=latent_dim,
            encode_dim=encode_dim
        )
        match processor:
            case "MPNN":
                self.processor=MPNN_Processor(
                    encode_dim=encode_dim,
                    latent_dim=latent_dim,
                    edge_dim=edge_dim,
                    aggr=aggr
                )
            case "GAT":
                self.processor=GAT_Processor(
                    encode_dim=encode_dim,
                    latent_dim=latent_dim,
                    edge_dim=edge_dim
                )
            case "GATv2":
                self.processor=GATv2_Processor(
                    encode_dim=encode_dim,
                    latent_dim=latent_dim,
                    edge_dim=edge_dim
                )
        self.decoder=Decoder(latent_dim=latent_dim)
        self.latent_dim=latent_dim

    def forward(self,
            r_traj:torch.Tensor,
            edge_index:torch.Tensor,
            edge_attr:torch.Tensor
        ):
        """
        Input: 
            r_traj: [B,max_bfs_len,N]
            bfs_mask: [B,max_bfs_len]
            edge_index
            edge_attr
        Return:
            pred_r_traj: [B,max_bfs_len-1,N]
        """
        batch_size,max_bfs_len,n_node=r_traj.size()
        device=r_traj.device

        ### set init latent feature
        h_0=torch.zeros((batch_size*n_node,self.latent_dim),dtype=torch.float32,device=device) # [B*N,latent_dim]
        pre_h=h_0

        ### compute pred_r_traj
        # r_traj[step]를 입력으로 사용해서 r_traj[step+1]을 예측
        pred_r_list=[]
        for step in range(max_bfs_len-1):
            # set current r
            r_t=r_traj[:,step ] # [B,N]
            r_t=r_t.reshape(batch_size*n_node,-1) # [B*N,1]

            # encode -> [B*N,encode_dim]
            z_t=self.encoder(
                x=r_t,
                h=pre_h
            ) 

            # process -> [B*N,latent_dim]
            h_t=self.processor(
                z=z_t,
                pre_h=pre_h,
                edge_index=edge_index,
                edge_attr=edge_attr
            )

            # decode -> # [B*N,1]
            y_t=self.decoder(
                z=z_t,
                h=h_t
            )
            pred_r=y_t.reshape(batch_size,n_node) # [B,N]
            pred_r_list.append(pred_r)

            # h_t를 다음 execution step으로 전달
            pre_h=h_t

        # get pred_r_traj
        pred_r_traj=torch.stack(pred_r_list,dim=1) # [B,max_bfs_len-1,N]
        return pred_r_traj