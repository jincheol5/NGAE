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
        self.latent_dim=latent_dim

    def forward(self,
            r_traj:torch.Tensor,
            d_traj:torch.Tensor,
            edge_index:torch.Tensor,
            edge_attr:torch.Tensor,
            mode:Literal["train","test"]="train"
        ):
        """
        각 step은 다음 state를 예측하며 padding은 각 loss의 mask로 제외한다.

        Input:
            r_traj: [B,max_bfs_len,N]
            d_traj: [B,max_bf_len,N]
            edge_index: [2,E], batch 전체의 global node id
            edge_attr: [E,edge_dim]
            mode: train은 graph별 50% teacher forcing, test는 이전 예측 사용
        Return:
            pred_r_traj: [B,max_bfs_len-1,N], raw logits
            pred_d_traj: [B,max_bf_len-1,N]
            edge_score_traj: [E,max_bf_len-1,1], raw logits
        """
        result={}
        edge_score_list=[]

        # processor 가중치는 공유하고 알고리즘별 잠재 상태는 독립적으로 유지
        for algorithm,traj in enumerate((r_traj,d_traj)):
            batch_size,n_step,n_node=traj.shape
            pre_h=edge_attr.new_zeros((batch_size*n_node,self.latent_dim))
            current=traj[:,0].to(dtype=edge_attr.dtype)
            predictions=[]

            for step in range(n_step-1):
                z_t=self.encoder[algorithm](
                    x=current.reshape(batch_size*n_node,1),
                    h=pre_h
                )
                h_t=self.processor(
                    z=z_t,
                    pre_h=pre_h,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )
                pred=self.decoder[algorithm](z=z_t,h=h_t).reshape(batch_size,n_node)
                predictions.append(pred)

                if algorithm==1:
                    edge_score_list.append(self.edge_score_network(
                        h=h_t,
                        edge_attr=edge_attr,
                        edge_index=edge_index
                    ))

                pre_h=h_t
                current=torch.sigmoid(pred) if algorithm==0 else pred
                if mode=="train":
                    teacher_mask=torch.rand(batch_size,1,device=traj.device)<0.5
                    current=torch.where(teacher_mask,traj[:,step+1],current)

            key="reachability" if algorithm==0 else "distance"
            result[key]=(
                torch.stack(predictions,dim=1) if predictions
                else edge_attr.new_empty((batch_size,0,n_node))
            )

        result["edge_score"]=(
            torch.stack(edge_score_list,dim=1) if edge_score_list
            else edge_attr.new_empty((edge_index.size(1),0,1))
        )
        return result


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
            edge_attr:torch.Tensor,
            mode:Literal["train","test"]="train"
        ):
        """
        Input: 
            r_traj: [B,max_bfs_len,N]
            edge_index
            edge_attr
            mode: 
                "train":
                    teacher forcing
                    각 graph마다 50% 확률로 ground truth, 50% 확률로 이전 step의 prediction 사용
                "test":
                    첫 step 이후 모든 입력으로 이전 step의 prediction 사용
        Return:
            pred_r_traj: [B,max_bfs_len-1,N]
        """
        batch_size,max_bfs_len,n_node=r_traj.size()
        device=r_traj.device

        ### set init latent feature
        h_0=torch.zeros(
            (batch_size*n_node,self.latent_dim),
            dtype=torch.float32,
            device=device
        ) # [B*N,latent_dim]
        pre_h=h_0

        ### compute pred_r_traj
        pred_r_list=[]

        # 첫 execution step의 입력은 항상 실제 initial state 사용
        current_r=r_traj[:,0] # [B,N]

        for step in range(max_bfs_len-1):
            r_t=current_r.reshape(batch_size*n_node,1) # [B*N,1]

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

            # 이전 step prediction을 다음 입력으로 사용하기 위해 sigmoid 적용
            pred_r_prob=torch.sigmoid(pred_r) # [B,N]
            match mode:
                case "train":
                    # set teacher forcing: graph마다 독립적으로 50% ground truth / 50% prediction
                    teacher_mask=(
                        torch.rand(batch_size,1,device=device)<0.5
                    )  # [B,1]
                    current_r=torch.where(
                        teacher_mask,
                        r_traj[:,step+1],
                        pred_r_prob
                    )  # [B,N]
                case "test":
                    # inference에서는 항상 이전 prediction 사용
                    current_r=pred_r_prob

        # get pred_r_traj
        pred_r_traj=torch.stack(pred_r_list,dim=1) # [B,max_bfs_len-1,N]
        return pred_r_traj

class NGAE_BF(nn.Module):
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
        self.edge_score_network=Edge_Wise_Scoring_Network(latent_dim=latent_dim,edge_dim=edge_dim)
        self.latent_dim=latent_dim

    def forward(self,
            d_traj:torch.Tensor,
            edge_index:torch.Tensor,
            edge_attr:torch.Tensor,
            mode:Literal["train","test"]="train"
        ):
        """
        Input: 
            d_traj: [B,max_bf_len,N]
            bf_mask: [B,max_bf_len], 호출 호환성을 유지하며 loss에서 padding 제외에 사용
            edge_index
            edge_attr
        Return:
            pred_d_traj: [B,max_bf_len-1,N]
            edge_score_traj: [E,max_bf_len-1,1], step별 다음 predecessor의 raw logits
        """
        batch_size,max_bf_len,n_node=d_traj.size()
        device=d_traj.device
        
        ### set init latent feature
        h_0=torch.zeros(
            (batch_size*n_node,self.latent_dim),
            dtype=torch.float32,
            device=device
        ) # [B*N,latent_dim]
        pre_h=h_0

        ### compute distance and edge score trajectories
        pred_d_list=[]
        edge_score_list=[]

        # 첫 execution step의 입력은 항상 실제 initial state 사용
        current_d=d_traj[:,0] # [B,N]
        for step in range(max_bf_len-1):
            d_t=current_d.reshape(batch_size*n_node,1) # [B*N,1]

            # encode -> [B*N,encode_dim]
            z_t=self.encoder(
                x=d_t,
                h=pre_h
            ) 

            # process -> [B*N,latent_dim]
            h_t=self.processor(
                z=z_t,
                pre_h=pre_h,
                edge_index=edge_index,
                edge_attr=edge_attr
            )

            # 현재 execution step에서 다음 predecessor의 점수 계산
            edge_score_list.append(self.edge_score_network(
                h=h_t,
                edge_attr=edge_attr,
                edge_index=edge_index
            )) # [E,1]

            # decode -> # [B*N,1]
            y_t=self.decoder(
                z=z_t,
                h=h_t
            )
            pred_d=y_t.reshape(batch_size,n_node) # [B,N]
            pred_d_list.append(pred_d)

            # h_t를 다음 execution step으로 전달
            pre_h=h_t

            match mode:
                case "train":
                    # set teacher forcing: graph마다 독립적으로 50% ground truth / 50% prediction
                    teacher_mask=(
                        torch.rand(batch_size,1,device=device)<0.5
                    )  # [B,1]
                    current_d=torch.where(
                        teacher_mask,
                        d_traj[:,step+1],
                        pred_d
                    )  # [B,N]
                case "test":
                    # inference에서는 항상 이전 prediction 사용
                    current_d=pred_d

        # get pred_d_traj
        pred_d_traj=torch.stack(pred_d_list,dim=1) # [B,max_bf_len-1,N]
        edge_score_traj=torch.stack(edge_score_list,dim=1) # [E,max_bf_len-1,1]
        return {
            "distance":pred_d_traj,
            "edge_score":edge_score_traj
        }
