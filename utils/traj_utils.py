import copy
import networkx as nx
import torch
from typing import Literal
from tqdm import tqdm
from torch_geometric.data import Data
from .graph_utils import GraphUtils,GraphAlgorithm

class TrajUtils:
    """
    src_data: graph의 각 node의 traj_data
    graph_data: graph의 모든 node들의 traj_data list 
    graph_data_list: graph_list 안의 모든 graph들의 graph_data list
    """
    @staticmethod
    def _compute_traj(
            graph:nx.Graph,
            source:int,
            graph_id:int,
            graph_type:Literal[
                "ladder",
                "grid",
                "tree",
                "erdos_renyi",
                "barabasi_albert",
                "community",
                "caveman"
            ]
        ):
        """
        Input:
            graph
            source
            graph_id
            graph_type
        Return:
            traj_data: PyG Data
        """
        ### Init setting for BFS
        BFS_graph=copy.deepcopy(graph)
        GraphUtils.init_node_state(graph=BFS_graph,source=source)
        BFS_Q=set().add(source)
        r_traj_list=[]
        r_traj_list.append(GraphUtils.get_node_state_tensor(graph=BFS_graph,state=f"r"))

        ### Init setting for Bellman-Ford
        BF_graph=copy.deepcopy(graph)
        GraphUtils.init_node_state(graph=BF_graph,source=source)
        BF_Q=set().add(source)
        d_traj_list=[]
        p_traj_list=[]
        d_traj_list.append(GraphUtils.get_node_state_tensor(graph=BF_graph,state=f"d"))
        p_traj_list.append(GraphUtils.get_node_state_tensor(graph=BF_graph,state=f"p"))

        ### Compute BFS, Bellman-Ford trajectory tensor
        while True:
            if not BFS_Q and not BF_Q: 
                break

            if BFS_Q:
                BFS_Q=GraphAlgorithm.compute_BFS_step(graph=BFS_graph,Q=BFS_Q)
                r_traj_list.append(GraphUtils.get_node_state_tensor(graph=BFS_graph,state=f"r"))

            if BF_Q:
                BF_Q=GraphAlgorithm.compute_BF_step(graph=BF_graph,Q=BF_Q)
                d_traj_list.append(GraphUtils.get_node_state_tensor(graph=BF_graph,state=f"d"))
                p_traj_list.append(GraphUtils.get_node_state_tensor(graph=BF_graph,state=f"p"))

        ### Convert to PyG Data
        # get edge_index, edge_attr
        edge_index=GraphUtils.get_edge_index_tensor(graph=graph)
        edge_attr=GraphUtils.get_edge_weight_tensor(graph=graph,edge_index=edge_index)

        # Init PyG Data
        data=Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=graph.number_of_nodes()
        )

        # Set meta data
        data.source=source
        data.graph_type=graph_type
        data.graph_id=graph_id

        # Set BFS, Bellman-Ford trajectory
        data.r=torch.stack(r_traj_list,dim=0) # [len_bfs_traj,N]
        data.d=torch.stack(d_traj_list,dim=0) # [len_bf_traj,N]
        data.p=torch.stack(p_traj_list,dim=0) # [len_bf_traj,N]
        return data

    @staticmethod
    def convert_graph_list(
            graph_list:list[nx.Graph],
            graph_type:Literal[
                "ladder",
                "grid",
                "tree",
                "erdos_renyi",
                "barabasi_albert",
                "community",
                "caveman"
            ]
        ):
        """
        Input:
            graph_list
            graph_type
        Return:
            graph_data_list: List[List[PyG Data]]
        """
        graph_data_list=[]
        for graph_id,graph in tqdm(
                enumerate(graph_list),
                total=len(graph_list),
                desc=f"Convert to graph_data_list..."
            ):
            graph_data=[]
            for src in graph.nodes():
                src_data=TrajUtils._compute_traj(
                    graph=graph,
                    source=src,
                    graph_id=graph_id,
                    graph_type=graph_type
                )
                graph_data.append(src_data)
            graph_data_list.append(graph_data)
        return graph_data_list
