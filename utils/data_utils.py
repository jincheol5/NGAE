import os
import pickle
import torch
from typing import Literal

BASE_PATH=os.path.join("..","data","neural_execution")
class DataUtils:
    @staticmethod
    def save_to_pickle(
            data,
            file_name:str,
            dir_type:Literal[
                "graph",
                "trajectory"
            ]
        ):
        file_name=file_name+".pkl"
        file_path=os.path.join(BASE_PATH,dir_type,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(data,f)
        print(f"Complete to save {file_name}!")

    @staticmethod
    def load_pickle(
            file_name:str,
            dir_type:Literal[
                "graph",
                "trajectory"
            ]
        ):
        file_name=file_name+".pkl"
        file_path=os.path.join(BASE_PATH,dir_type,file_name)
        with open(file_path,'rb') as f:
            data=pickle.load(f)
        return data

    @staticmethod
    def save_to_pt(
            data,
            file_name:str,
            dir_type:Literal[
                "graph",
                "trajectory"
            ],
            mode:Literal[
                "train",
                "val",
                "test"
            ]
        ):
        file_name=file_name+".pt"
        file_path=os.path.join(BASE_PATH,dir_type,mode,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        torch.save(data,file_path)
        print(f"Complete to save {file_name}!")

    @staticmethod
    def load_pt(
            file_name:str,
            dir_type:Literal[
                "graph",
                "trajectory"
            ],
            mode:Literal[
                "train",
                "val",
                "test"
            ]
        ):
        file_name=file_name+".pt"
        file_path=os.path.join(BASE_PATH,dir_type,mode,file_name)
        data=torch.load(
            file_path,
            weights_only=False
        )
        return data