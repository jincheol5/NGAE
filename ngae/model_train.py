import os
import numpy as np
import wandb
import torch
from tqdm import tqdm
from .metrics import Metrics
from .model import NGAE_BFS,NGAE_BF
from .data_utils import DataUtils

class ModelTrainer:
    @staticmethod
    def train(model,train_data_loader,val_data_loader_dict,config: dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])

        for epoch in tqdm(range(config['epochs']),desc=f"Training {config['task']}..."):
            """
            epoch loss list for wandb
            """
            epoch_algo_loss=[]
            epoch_p_loss=[]
            epoch_tau_loss=[]
            epoch_total_loss=[]

            """
            model train
            """
            model.train()
            for batch in tqdm(train_data_loader,desc=f"Epoch {epoch}..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                match config['task']:
                    case 'bfs':
                        algo_trajectory=batch.bfs # [seq_len,N,1]
                        tau_seq_label=batch.bfs_tau # [seq_len-1,1]
                    case 'bf':
                        algo_trajectory=batch.bf # [seq_len,N,1]
                        p_idx_trajectory=batch.p_idx # [seq_len,N,1]
                        tau_seq_label=batch.bf_tau # [seq_len-1,1]

                output=model(algo_trajectory=algo_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,mode='train')
                y_seq=output['y'] # [seq_len-1,N,1]
                tau_seq=output['tau'] # [seq_len-1,1]
                if config['task']=='bf':
                    edge_score_seq=output['edge_score'] # [seq_len-1,E,1]

                total_loss=torch.zeros((),device=device)
                match config['task']:
                    case 'bfs':
                        y_seq_loss=Metrics.compute_BFS_seq_loss(logit=y_seq,label=algo_trajectory[1:])
                        tau_seq_loss=Metrics.compute_tau_seq_loss(logit=tau_seq,label=tau_seq_label)
                        total_loss=y_seq_loss+tau_seq_loss

                        # wandb
                        epoch_algo_loss.append(y_seq_loss)
                        epoch_tau_loss.append(tau_seq_loss)
                        epoch_total_loss.append(total_loss)
                    case 'bf':
                        y_seq_loss=Metrics.compute_BF_seq_loss(logit=y_seq,label=algo_trajectory[1:])
                        p_seq_loss=Metrics.compute_predecessor_seq_loss(logit=edge_score_seq,label=p_idx_trajectory[1:],edge_index=batch.edge_index)
                        tau_seq_loss=Metrics.compute_tau_seq_loss(logit=tau_seq,label=tau_seq_label)
                        total_loss=y_seq_loss+p_seq_loss+tau_seq_loss

                        # wandb
                        epoch_algo_loss.append(y_seq_loss)
                        epoch_p_loss.append(p_seq_loss)
                        epoch_tau_loss.append(tau_seq_loss)
                        epoch_total_loss.append(total_loss)
                """
                back propagation
                """
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            """
            wandb log
            """
            match config['task']:
                case 'bfs':
                    wandb.log({
                        'bfs_loss':torch.stack(epoch_algo_loss).mean(),
                        'tau_loss':torch.stack(epoch_tau_loss).mean(),
                        'total_loss':torch.stack(epoch_total_loss).mean()
                    },step=epoch)
                case 'bf':
                    wandb.log({
                        'bf_loss':torch.stack(epoch_algo_loss).mean(),
                        'p_loss':torch.stack(epoch_p_loss).mean(),
                        'tau_loss':torch.stack(epoch_tau_loss).mean(),
                        'total_loss':torch.stack(epoch_total_loss).mean()
                    },step=epoch)
            """
            validate
            """
            for val_graph_type,val_data_loader in val_data_loader_dict.items():
                ModelTrainer.test(model=model,graph_type=val_graph_type,data_loader=val_data_loader,config=config)
        return model

    @staticmethod
    def test(model,graph_type,data_loader,config):
        if config['mode']=="test":
            match config['model_name']:
                case 'NGAE_bfs':
                    model=NGAE_BFS(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
                case 'NGAE_bf':
                    model=NGAE_BF(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
                case 'NGAE':
                    pass
            model=DataUtils.DataLoader.load_model_parameter(model=model,model_name=config['model_name'])
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        match config['task']:
            case 'bfs':
                y_step_acc_list=[]
                y_last_acc_list=[]
                tau_step_acc_list=[]
                tau_last_acc_list=[]
            case 'bf':
                p_step_acc_list=[]
                p_last_acc_list=[]
                tau_step_acc_list=[]
                tau_last_acc_list=[]

        with torch.no_grad():
            for batch in tqdm(data_loader,desc=f"{config['mode']} {graph_type} graph..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                match config['task']:
                    case 'bfs':
                        algo_trajectory=batch.bfs # [seq_len,N,1]
                        tau_seq_label=batch.bfs_tau # [seq_len-1,1]
                    case 'bf':
                        algo_trajectory=batch.bf # [seq_len,N,1]
                        p_idx_trajectory=batch.p_idx # [seq_len,N,1]
                        tau_seq_label=batch.bf_tau # [seq_len-1,1]

                output=model(algo_trajectory=algo_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,mode='test')
                y_seq=output['y'] # [seq_len-1,N,1]
                tau_seq=output['tau'] # [seq_len-1,1]
                if config['task']=='bf':
                    edge_score_seq=output['edge_score'] # [seq_len-1,E,1]

                """
                compute acc
                """
                match config['task']:
                    case 'bfs':
                        batch_y_step_acc,batch_y_last_acc=Metrics.compute_BFS_seq_acc(logit=y_seq,label=algo_trajectory[1:])
                        batch_tau_step_acc,batch_tau_last_acc=Metrics.compute_tau_seq_acc(logit=tau_seq,label=tau_seq_label)
                        y_step_acc_list.append(batch_y_step_acc)
                        y_last_acc_list.append(batch_y_last_acc)
                        tau_step_acc_list.append(batch_tau_step_acc)
                        tau_last_acc_list.append(batch_tau_last_acc)
                    case 'bf':
                        batch_p_step_acc,batch_p_last_acc=Metrics.compute_predecessor_seq_acc(logit=edge_score_seq,label=p_idx_trajectory[1:],edge_index=batch.edge_index)
                        batch_tau_step_acc,batch_tau_last_acc=Metrics.compute_tau_seq_acc(logit=tau_seq,label=tau_seq_label)
                        p_step_acc_list.append(batch_p_step_acc)
                        p_last_acc_list.append(batch_p_last_acc)
                        tau_step_acc_list.append(batch_tau_step_acc)
                        tau_last_acc_list.append(batch_tau_last_acc)
        
        match config['task']:
            case 'bfs':
                y_step_acc=np.mean(y_step_acc_list)
                y_last_acc=np.mean(y_last_acc_list)
                tau_step_acc=np.mean(tau_step_acc_list)
                tau_last_acc=np.mean(tau_last_acc_list)

                print(f"{config['mode']} {graph_type} graph BFS step acc: {y_step_acc} last acc: {y_last_acc}")
                print(f"{config['mode']} {graph_type} graph tau step acc: {tau_step_acc} last acc: {tau_last_acc}")
            case 'bf':
                p_step_acc=np.mean(p_step_acc_list)
                p_last_acc=np.mean(p_last_acc_list)
                tau_step_acc=np.mean(tau_step_acc_list)
                tau_last_acc=np.mean(tau_last_acc_list)

                print(f"{config['mode']} {graph_type} graph predecessor step acc: {p_step_acc} last acc: {p_last_acc}")
                print(f"{config['mode']} {graph_type} graph tau step acc: {tau_step_acc} last acc: {tau_last_acc}")