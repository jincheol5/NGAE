import os
import numpy as np
import pickle
import gzip
import networkx as nx
import torch
from tqdm import tqdm
from typing_extensions import Literal
from graph_utils import GraphUtils
from torch_geometric.utils import from_networkx,sort_edge_index

class DataUtils:
    class DataLoader:
        dataset_path=os.path.join('..','data','ngae')
        @staticmethod
        def save_to_pickle(data,file_name: str,dir_type: Literal['graph','test','train','val']):
            file_name=file_name+".pkl.gz"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,dir_type,file_name)
            with gzip.open(file_path,'wb') as f:
                pickle.dump(data,f)
            print(f"Save {file_name} (compressed with gzip)")
        
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
            GraphUtils.GraphManager.initialize_node_attr_for_BFS(graph=graph,source_id=source_id)
            GraphUtils.GraphManager.initialize_node_attr_for_BF(graph=graph,source_id=source_id)
