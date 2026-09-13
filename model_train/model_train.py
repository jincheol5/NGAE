import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from utils import EarlyStopper

class ModelTrainer:
    @staticmethod
    def train(
            model:nn.Module,
            train_dict:dict[str,list[list[Data]]],
            val_loader:DataLoader,
            **kwargs
        ):
        """
        Set GPU, Optimizer
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)

        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )

        """
        Set Early Stopper
        """
        if kwargs["early_stop"]:
            early_stop=EarlyStopper(patience=kwargs["patience"])

        """
        Model train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            for batch_data in tqdm(train_loader,desc=f"Training epoch {epoch+1}..."):
                """
                """