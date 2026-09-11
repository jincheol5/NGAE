import copy
import networkx as nx
from typing import Literal
from .graph_utils import GraphUtils,GraphAlgorithm

class TrainUtils:
    @staticmethod
    def _convert_traj_to_pyg_data(
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
        """
        # Init setting for BFS
        bfs_graph=copy.deepcopy(graph)
        GraphUtils.init_node_state(graph=bfs_graph,source=source)
        bfs_Q=set().add(source)
        r_traj_tensor_list=[]
        r_traj_tensor_list.append(GraphUtils.get_node_state_tensor(graph=bfs_graph,state=f"r"))

        # Init setting for Bellman-Ford
        bf_graph=copy.deepcopy(graph)
        GraphUtils.init_node_state(graph=bf_graph,source=source)
        bf_Q=set().add(source)
        d_traj_tensor_list=[]
        p_traj_tensor_list=[]
        d_traj_tensor_list.append(GraphUtils.get_node_state_tensor(graph=bf_graph,state=f"d"))
        p_traj_tensor_list.append(GraphUtils.get_node_state_tensor(graph=bf_graph,state=f"p"))