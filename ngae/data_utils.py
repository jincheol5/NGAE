import os
import numpy as np
import pickle
import gzip
import networkx as nx
import torch
from tqdm import tqdm
from typing_extensions import Literal
from graph_utils import GraphUtils
from torch_geometric.data import Data

class DataUtils:
    class DataLoader:
        dataset_path=os.path.join('..','data','ngae')
        @staticmethod
        def save_to_pickle(data,file_name: str,dir_type: Literal['graph','test','train','val']):
            file_name=file_name+".pkl.gz"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,dir_type,file_name)
            with gzip.open(file_path,'wb') as f:
                pickle.dump(data,f)
            print(f"Save {file_name} (Compressed with gzip)")
        
        @staticmethod
        def load_from_pickle(file_name: str,dir_type: Literal['graph','test','train','val']):
            file_name=file_name+".pkl.gz"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,dir_type,file_name)
            with gzip.open(file_path,'rb') as f:
                data=pickle.load(f)
            print(f"Load {file_name} (Decompressed with gzip)")
            return data
    
    class DataProcessor:
        @staticmethod
        def algo_trajectory_to_PyG_Data(graph: nx.DiGraph,graph_type: str="basic",graph_num: int=0,source_id: int=0):
            """
            initialize
            """
            bfs_Q=GraphUtils.GraphAlgorithm.compute_BFS_step(graph=graph,source_id=source_id,init=True)
            bfs_trajectory_tensor_list=[]
            bfs_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bfs'))

            bf_Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=source_id,init=True)
            bf_trajectory_tensor_list=[]
            p_trajectory_tensor_list=[]
            p_idx_trajectory_tensor_list=[]
            bf_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bf'))
            p_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p'))
            p_idx_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p_idx'))

            """
            compute BFS, Bellman-Ford trajectory: [seq_len,num_nodes,1]
            """
            while(True):
                if not bfs_Q and not bf_Q:
                    break
                if bfs_Q:
                    bfs_Q=GraphUtils.GraphAlgorithm.compute_BFS_step(graph=graph,source_id=source_id,Q=bfs_Q)
                    bfs_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bfs'))
                if bf_Q:
                    bf_Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=source_id,Q=bf_Q)
                    bf_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bf'))
                    p_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p'))
                    GraphUtils.GraphManager.remap_node_predecessor_to_sorted_index(graph=graph)
                    p_idx_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p_idx'))
            
            bfs_trajectory=torch.stack(bfs_trajectory_tensor_list,dim=0)
            bf_trajectory=torch.stack(bf_trajectory_tensor_list,dim=0)
            p_trajectory=torch.stack(p_trajectory_tensor_list,dim=0)
            p_idx_trajectory=torch.stack(p_idx_trajectory_tensor_list,dim=0)

            """
            compute tau: [seq_len,1]
            """
            bfs_tau=torch.cat([torch.ones(len(bfs_trajectory_tensor_list)-1,1),torch.zeros(1,1)],dim=0)
            bf_tau=torch.cat([torch.ones(len(bf_trajectory_tensor_list)-1,1),torch.zeros(1,1)],dim=0)

            """
            convert to PyG Data
            """
            edge_index=GraphUtils.GraphManager.get_sorted_edge_index_tensor(graph=graph)
            edge_attr=GraphUtils.GraphManager.get_edge_w_tensor(graph=graph,edge_index=edge_index)
            data=Data(edge_index=edge_index,edge_attr=edge_attr,num_nodes=graph.number_of_nodes())

            # meta data
            data.source_id=source_id
            data.graph_type=graph_type
            data.graph_num=graph_num

            # BFS
            data.bfs=bfs_trajectory
            data.bfs_tau=bfs_tau

            # Bellman-ford
            data.bf=bf_trajectory
            data.p=p_trajectory
            data.p_idx=p_idx_trajectory
            data.bf_tau=bf_tau

            return data