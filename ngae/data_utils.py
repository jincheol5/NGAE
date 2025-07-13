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
            print(f"Save {file_name} (Compressed with gzip)")
        
        @staticmethod
        def load_from_pickle(file_name: str,dir_type: Literal['graph','test','train','val']):
            file_name=file_name+".pkl.gz"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,dir_type,file_name)
            with gzip.open(file_path,'rb') as f:
                data=pickle.load(f)
            print(f"Load {file_name} (Decompressed with gzip)")
            return data

        @staticmethod
        def save_data_list(data_list_dict: dict,file_name: str,dir_type: Literal['graph','test','train','val']):
            for key,value in tqdm(data_list_dict.items(),desc=f"Save {file_name} data list..."):
                match key:
                    case 'all':
                        DataUtils.DataLoader.save_to_pickle(data=value,file_name=file_name,dir_type=dir_type)
                    case 'ladder':
                        DataUtils.DataLoader.save_to_pickle(data=value,file_name=file_name+"_ladder",dir_type=dir_type)
                    case 'grid':
                        DataUtils.DataLoader.save_to_pickle(data=value,file_name=file_name+"_grid",dir_type=dir_type)
                    case 'tree':
                        DataUtils.DataLoader.save_to_pickle(data=value,file_name=file_name+"_tree",dir_type=dir_type)
                    case 'erdos_renyi':
                        DataUtils.DataLoader.save_to_pickle(data=value,file_name=file_name+"_erdos_renyi",dir_type=dir_type)
                    case 'barabasi_albert':
                        DataUtils.DataLoader.save_to_pickle(data=value,file_name=file_name+"_barabasi_albert",dir_type=dir_type)
                    case 'community':
                        DataUtils.DataLoader.save_to_pickle(data=value,file_name=file_name+"_community",dir_type=dir_type)
                    case 'caveman':
                        DataUtils.DataLoader.save_to_pickle(data=value,file_name=file_name+"_caveman",dir_type=dir_type)
        
        @staticmethod
        def save_model_parameter(model,model_name="NGAE"):
            file_name=model_name+".pt"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,"inference",file_name)
            torch.save(model.state_dict(),file_path)
            print(f"Save {model_name} model parameter")
        
        @staticmethod
        def load_model_parameter(model,model_name="NGAE"):
            file_name=model_name+".pt"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,"inference",file_name)
            model.load_state_dict(torch.load(file_path))
            return model