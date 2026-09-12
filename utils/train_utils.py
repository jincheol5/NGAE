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
            label_r_traj[valid_mask].float()
        )
        return loss