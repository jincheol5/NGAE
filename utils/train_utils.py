import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,Batch

class TrainUtils:
    @staticmethod
    def custom_collate_fn(data_list:list[Data]):
        """
        DataLoader가 여러 sample을 하나의 batch로 묶을 때 "어떻게 합칠지"를 정의하는 함수
        
        PyG Data list를 하나의 batch로 변환.
        - edge_index: PyG Batch가 graph별 node offset을 자동 적용
        - edge_attr: 모든 graph의 edge feature를 concatenate
        - r: [B,max_bfs_len,N]
        - d: [B,max_bf_len,N]
        - p: [B,max_bf_len,N]
            predecessor node id에 graph별 global node offset 적용

        Assumption:
            - batch 내 모든 graph의 num_nodes가 동일

        Return:
            batch.r: [B,max_bfs_len,N]
            batch.d: [B,max_bf_len,N]
            batch.p: [B,max_bf_len,N], global node indexing으로 변환됨
            batch.bfs_mask: [B,max_bfs_len]
            batch.bf_mask: [B,max_bf_len]
        """
        batch_size=len(data_list)
        n_node=data_list[0].num_nodes
        max_bfs_len=max(
            data.r.size(0)
            for data in data_list
        )
        max_bf_len=max(
            data.d.size(0)
            for data in data_list
        )

        ### init padding tensor
        batch_r=torch.zeros((batch_size,max_bfs_len,n_node),dtype=torch.float32)
        batch_d=torch.zeros((batch_size,max_bf_len,n_node),dtype=torch.float32)
        batch_p=torch.full((batch_size,max_bf_len,n_node),fill_value=-1,dtype=torch.long)

        # trajectory step mask
        bfs_mask=torch.zeros((batch_size,max_bfs_len),dtype=torch.bool)
        bf_mask=torch.zeros((batch_size,max_bf_len),dtype=torch.bool)

        graph_list=[]
        node_offset=0
        for idx,data in enumerate(data_list):
            bfs_len=data.r.size(0)
            bf_len=data.d.size(0)

            # BFS reachability trajectory
            batch_r[idx,:bfs_len]=data.r
            bfs_mask[idx,:bfs_len]=True

            # Bellman-Ford distance trajectory
            batch_d[idx,:bf_len]=data.d
            bf_mask[idx,:bf_len]=True

            # Bellman-Ford predecessor trajectory
            p=data.p.clone()
            p+=node_offset
            batch_p[idx,:bf_len]=p

            # PyG가 r/d/p를 자동 concat하지 않도록 제거
            graph_data=data.clone()
            del graph_data.r
            del graph_data.d
            del graph_data.p
            graph_list.append(graph_data)

            # update node_offset
            node_offset+=data.num_nodes

        # PyG batching
        batch=Batch.from_data_list(graph_list)

        # batch에 batch_traj 추가
        batch.r=batch_r # [B,max_bfs_len,N]
        batch.d=batch_d # [B,max_bf_len,N]
        batch.p=batch_p # [B,max_bf_len,N]
        batch.bfs_mask=bfs_mask # [B,max_bfs_len]
        batch.bf_mask=bf_mask # [B,max_bf_len]
        return batch

    @staticmethod
    def sampling_k_source_traj_per_graph(
            graph_data_list_dict:dict[str,list[list[Data]]],
            k:int=1
        )->list[Data]:
        """
        각 graph마다 source trajectory를 최대 k개 랜덤 샘플링

        Input:
            graph_data_list_dict: dict
                graph_type: str
                graph_data_list: list
                    graph_data: list
                        src_data: PyG Data
        Return:
            sampled_data_list: list of PyG Data
        """
        sampled_data_list=[]
        for _,graph_data_list in graph_data_list_dict.items():
            for graph_data in graph_data_list:
                n_sample=min(k,len(graph_data))
                sampled_data=random.sample(
                    graph_data,
                    k=n_sample
                )
                sampled_data_list.extend(sampled_data)
        return sampled_data_list

    @staticmethod
    def compute_reachability_loss(
            r_traj:torch.Tensor,
            pred_r_traj:torch.Tensor,
            bfs_mask:torch.Tensor
        ):
        """
        현재 state와 다음 state가 둘 다 실제 데이터에 존재하는 transition만 loss에 포함

        Input:
            r_traj: [B,max_bfs_len,N]
            pred_r_traj: [B,max_bfs_len-1,N]
            bfs_mask: [B,max_bfs_len]
        """
        # get label
        label_r_traj=r_traj[:,1:] # [B,max_bfs_len-1,N]

        # 현재 step과 다음 step이 모두 존재하는 transition만 사용
        valid_mask=(bfs_mask[:,:-1] & bfs_mask[:,1:]) # [B,max_bfs_len-1]

        # graph-level step mask -> node-level mask
        valid_mask=valid_mask.unsqueeze(-1).expand_as(label_r_traj) # [B,max_bfs_len-1,N]

        # valid trajectory에 대해서만 BCE loss 계산
        loss=F.binary_cross_entropy_with_logits(
            pred_r_traj[valid_mask],
            label_r_traj[valid_mask]
        )
        return loss

    @staticmethod
    def compute_distance_loss(
            d_traj: torch.Tensor,
            pred_d_traj: torch.Tensor,
            bf_mask: torch.Tensor
        ):
        """
        현재 state와 다음 state가 둘 다 실제 데이터에 존재하는 transition만 loss에 포함

        Input:
            d_traj: [B,max_bf_len,N]
            pred_d_traj: [B,max_bf_len-1,N]
            bf_mask: [B,max_bf_len]
        """
        # get label
        label_d_traj=d_traj[:,1:] # [B,max_bf_len-1,N]

        # 현재 step과 다음 step이 모두 존재하는 transition만 사용
        valid_mask=(bf_mask[:,:-1] & bf_mask[:,1:]) # [B,max_bf_len-1]

        # graph-level step mask -> node-level mask
        valid_mask=valid_mask.unsqueeze(-1).expand_as(label_d_traj) # [B,max_bf_len-1,N]

        # valid trajectory에 대해서만 MSE loss 계산
        loss=F.mse_loss(
            pred_d_traj[valid_mask],
            label_d_traj[valid_mask]
        )
        return loss

class EarlyStopper:
    def __init__(self,
            patience:int=1
        ):
        self.patience=patience
        self.patience_count=0
        self.best_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,
            val_loss:float,
            model:torch.nn.Module
        ):
        # val_loss가 NaN, Inf이면 즉시 early stop
        if not np.isfinite(val_loss): 
            print("Loss is NaN or Inf!")
            self.early_stop=True
            if self.best_state is not None:
                model.load_state_dict(self.best_state)
            return model

        # 첫 번째 validation에서는 비교할 이전 best가 없으므로 현재 loss와 모델을 그대로 best로 저장
        if self.best_state is None: 
            self.best_loss=val_loss
            self.best_state={
                key: value.detach().clone()
                for key,value in model.state_dict().items()
            }
            return model

        # val_loss가 개선 되지 않은 경우
        if self.best_loss<=val_loss: 
            self.patience_count+=1
            if self.patience<=self.patience_count:
                self.early_stop=True
                model.load_state_dict(self.best_state)
            return model

        # val_loss가 개선 된 경우
        self.patience_count=0
        self.best_loss=val_loss
        self.best_state={
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        return model