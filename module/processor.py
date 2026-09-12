import torch
import torch.nn as nn
from typing_extensions import Literal
from torch_geometric.nn import MessagePassing,GATConv,GATv2Conv

"""
<< Note >>
- GAT-based의 경우 add_self_loops 설정 후 self-loop의 edge_attr = 0.0으로 설정
- Processor => 인접 노드가 없는 경우 이전 h를 그대로 사용하도록 설정
- Processor에서 ReLU를 적용 (Google DeepMind CLRS github 공식코드 참고)
"""
class MPNN_Processor(MessagePassing):
    def __init__(self,
            encode_dim:int,
            latent_dim:int,
            edge_dim:int,
            aggr:Literal["max","mean","add"]
            ):
        super().__init__(aggr=aggr)
        self.message_linear=nn.Linear(
            in_features=encode_dim+encode_dim+edge_dim,
            out_features=latent_dim
        ) 
        self.update_linear=nn.Linear(
            in_features=encode_dim+latent_dim,
            out_features=latent_dim
        )
        self.relu=nn.ReLU()

    def message(self,z_i,z_j,edge_attr):
        message=self.message_linear(torch.cat([z_i,z_j,edge_attr],dim=-1))
        return message

    def update(self,aggr_out,z):
        # tensor 안의 NaN, +inf, -inf 값을 지정한 숫자로 바꿔주는 함수
        # aggregation 단계에서 incoming edge가 하나도 없는 node의 경우 +inf, -inf 값이 나올 수 있음
        aggr_out=torch.nan_to_num(aggr_out,neginf=0.0,posinf=0.0) 
        h=self.update_linear(torch.cat([z,aggr_out],dim=-1))
        h=self.relu(h)
        return h

    def forward(self,
            z:torch.Tensor,
            pre_h: torch.Tensor,
            edge_index:torch.Tensor,
            edge_attr:torch.Tensor
        ):
        h=self.propagate(
            edge_index=edge_index,
            z=z,
            edge_attr=edge_attr
        )

        # 원본 그래프 기준 incoming edge 존재 여부
        tar=edge_index[1]
        has_incoming_edge=torch.zeros(
            z.size(0),
            dtype=torch.bool,
            device=z.device
        )
        has_incoming_edge[tar]=True

        # 원본 그래프에서 incoming edge가 없는 node는 pre_h 유지
        h=torch.where(has_incoming_edge.unsqueeze(-1),h,pre_h)
        return h

class GAT_Processor(nn.Module):
    def __init__(self,
            encode_dim:int,
            latent_dim:int,
            edge_dim:int
        ):
        super().__init__()
        self.gat=GATConv(
            in_channels=encode_dim,
            out_channels=latent_dim,
            heads=3,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=True,
            fill_value=0.0 # self-loop의 edge_attr 설정
        )
        self.relu=nn.ReLU()

    def forward(self,
            z:torch.Tensor,
            pre_h:torch.Tensor,
            edge_index:torch.Tensor,
            edge_attr:torch.Tensor
        ):
        h=self.gat(z,edge_index,edge_attr)
        h=self.relu(h)

        # 원본 그래프 기준 incoming edge 존재 여부
        tar=edge_index[1]
        has_incoming_edge=torch.zeros(
            z.size(0),
            dtype=torch.bool,
            device=z.device
        )
        has_incoming_edge[tar]=True

        # 원본 그래프에서 incoming edge가 없는 node는 pre_h 유지
        h=torch.where(has_incoming_edge.unsqueeze(-1),h,pre_h)
        return h


class GATv2_Processor(nn.Module):
    def __init__(self,
            encode_dim:int,
            latent_dim:int,
            edge_dim:int
        ):
        super().__init__()
        self.gat=GATv2Conv(
            in_channels=encode_dim,
            out_channels=latent_dim,
            heads=3,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=True,
            fill_value=0.0 # self-loop의 edge_attr 설정
        )
        self.relu=nn.ReLU()

    def forward(self,
            z:torch.Tensor,
            pre_h:torch.Tensor,
            edge_index:torch.Tensor,
            edge_attr:torch.Tensor
        ):
        h=self.gat(z,edge_index,edge_attr)
        h=self.relu(h)

        # 원본 그래프 기준 incoming edge 존재 여부
        tar=edge_index[1]
        has_incoming_edge=torch.zeros(
            z.size(0),
            dtype=torch.bool,
            device=z.device
        )
        has_incoming_edge[tar]=True

        # 원본 그래프에서 incoming edge가 없는 node는 pre_h 유지
        h=torch.where(has_incoming_edge.unsqueeze(-1),h,pre_h)
        return h