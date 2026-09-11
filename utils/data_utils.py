import os
import pickle
from typing import Literal

BASE_PATH=os.path.join("..","data","neural_execution")
class DataUtils:
    @staticmethod
    def save_to_pickle(
            data,
            file_name:str,
            dir_type:Literal[
                "graph",
                ""
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
                ""
            ]
        ):
        file_name=file_name+".pkl"
        file_path=os.path.join(BASE_PATH,dir_type,file_name)
        with open(file_path,'rb') as f:
            data=pickle.load(f)
        print(f"Complete to load {file_name}!")
        return data