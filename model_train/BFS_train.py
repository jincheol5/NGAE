import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from utils import TrainUtils,Metric,EarlyStopper

class BFSTrainer:
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
        Train model 
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            ### get train_loader
            train_data_list=TrainUtils.sampling_k_source_traj_per_graph(
                graph_data_list_dict=train_dict,
                k=1
            )
            train_loader=DataLoader(
                dataset=train_data_list,
                batch_size=kwargs["batch_size"],
                shuffle=True,
                collate_fn=TrainUtils.custom_collate_fn
            )
            model.train()
            for batch_data in tqdm(train_loader,desc=f"Training epoch {epoch+1}..."):
                ### get batch data
                r_traj=batch_data.r
                edge_index=batch_data.edge_index
                edge_attr=batch_data.edge_attr
                bfs_mask=batch_data.bfs_mask
                r_traj=r_traj.to(device)
                edge_index=edge_index.to(device)
                edge_attr=edge_attr.to(device)
                bfs_mask=bfs_mask.to(device)

                ### Forward
                pred_r_traj=model(
                    r_traj=r_traj,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    mode="train"
                )

                ### Loss
                loss=TrainUtils.compute_reachability_loss(
                    r_traj=r_traj,
                    pred_r_traj=pred_r_traj,
                    bfs_mask=bfs_mask
                )

                ### Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            """
            Validate model
            """
            val_result=BFSTrainer.validate(
                model=model,
                val_loader=val_loader,
                **kwargs
            )
            val_acc=val_result["acc"]
            print(f"Validate ACC: {val_acc}")

            """
            Check Early Stop
            """
            val_loss=val_result["loss"]
            print(f"{epoch+1} epoch Validate Loss: {val_loss}")
            if kwargs["early_stop"]:
                pre_model=early_stop(
                    val_loss=val_loss,
                    model=model
                )
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stop in epoch {epoch+1}")
                    break
        return model

    @staticmethod
    def validate(
            model:nn.Module,
            val_loader:DataLoader,
            **kwargs
        ):
        """
        Set GPU
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.eval()

        """
        Compute validate loss and acc
        """
        loss_list=[]
        acc_list=[]
        with torch.no_grad():
            for batch_data in tqdm(val_loader,desc=f"Validate..."):
                ### get batch data
                r_traj=batch_data.r
                edge_index=batch_data.edge_index
                edge_attr=batch_data.edge_attr
                bfs_mask=batch_data.bfs_mask
                r_traj=r_traj.to(device)
                edge_index=edge_index.to(device)
                edge_attr=edge_attr.to(device)
                bfs_mask=bfs_mask.to(device)

                ### Forward
                pred_r_traj=model(
                    r_traj=r_traj,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    mode="test"
                )

                ### Loss
                loss=TrainUtils.compute_reachability_loss(
                    r_traj=r_traj,
                    pred_r_traj=pred_r_traj,
                    bfs_mask=bfs_mask
                )
                loss_list.append(loss)

                ### Accuracy
                acc=Metric.compute_reachability_accuracy(
                    r_traj=r_traj,
                    pred_r_traj=pred_r_traj,
                    bfs_mask=bfs_mask
                )
                acc_list.append(acc)
        return {
            "loss":torch.stack(loss_list).mean().item(),
            "acc":sum(acc_list)/len(acc_list)
        }

    @staticmethod
    def evaluate(
            model:nn.Module,
            test_loader:DataLoader,
            **kwargs
        ):
        """
        Set GPU
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.eval()

        """
        Compute evaluate acc
        """
        acc_list=[]
        with torch.no_grad():
            for batch_data in tqdm(test_loader,desc=f"Validate..."):
                ### get batch data
                r_traj=batch_data.r
                edge_index=batch_data.edge_index
                edge_attr=batch_data.edge_attr
                bfs_mask=batch_data.bfs_mask
                r_traj=r_traj.to(device)
                edge_index=edge_index.to(device)
                edge_attr=edge_attr.to(device)
                bfs_mask=bfs_mask.to(device)

                ### Forward
                pred_r_traj=model(
                    r_traj=r_traj,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    mode="test"
                )

                ### Accuracy
                acc=Metric.compute_reachability_accuracy(
                    r_traj=r_traj,
                    pred_r_traj=pred_r_traj,
                    bfs_mask=bfs_mask
                )
                acc_list.append(acc)
        return {
            "acc":sum(acc_list)/len(acc_list)
        }