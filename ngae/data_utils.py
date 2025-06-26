import os
import numpy as np
import pickle
import gzip
import networkx as nx
import torch
from tqdm import tqdm
from typing_extensions import Literal

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
        pass