import os
import numpy as np
import wandb
import torch
from tqdm import tqdm
from .metrics import Metrics
from .model import NGAE_MPNN_BFS,NGAE_MPNN_BF,NGAE_MPNN,NGAE_GAT_BFS,NGAE_GAT_BF,NGAE_GAT
from .model_train_utils import EarlyStopping
from .data_utils import DataUtils

class ModelTrainer:
    @staticmethod
    def train(model,train_data_loader,val_data_loader_dict,config: dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])
        early_stop=EarlyStopping(patience=config['patience'])

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
            for batch in tqdm(train_data_loader,desc=f"Epoch {epoch+1}..."):
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
            Early stopping
            """
            algo_loss=torch.stack(epoch_algo_loss).mean()
            pre_model=early_stop(val_loss=algo_loss,model=model)
            if early_stop.early_stop:
                model=pre_model
                print(f"Early Stopping in epoch {epoch+1}")
                break

            """
            wandb log
            """
            if config['wandb']:
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
    def train_simultaneously(model,train_data_loader,val_data_loader_dict,config: dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])
        early_stop=EarlyStopping(patience=config['patience'])

        for epoch in tqdm(range(config['epochs']),desc=f"Training {config['task']}..."):
            """
            epoch loss list for wandb
            """
            epoch_bfs_loss=[]
            epoch_bf_loss=[]
            epoch_p_loss=[]
            epoch_bfs_tau_loss=[]
            epoch_bf_tau_loss=[]
            epoch_total_loss=[]

            """
            model train
            """
            model.train()
            for batch in tqdm(train_data_loader,desc=f"Epoch {epoch+1}..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                bfs_trajectory=batch.bfs
                bf_trajectory=batch.bf 
                p_idx_trajectory=batch.p_idx
                bfs_tau_seq_label=batch.bfs_tau 
                bf_tau_seq_label=batch.bf_tau

                output=model(bfs_trajectory=bfs_trajectory,bf_trajectory=bf_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,mode='train')
                bfs_y_seq=output['bfs_y']
                bf_y_seq=output['bf_y']
                edge_score_seq=output['edge_score']   
                bfs_tau_seq=output['bfs_tau']
                bf_tau_seq=output['bf_tau']  
                
                total_loss=torch.zeros((),device=device)
                bfs_seq_loss=Metrics.compute_BFS_seq_loss(logit=bfs_y_seq,label=bfs_trajectory[1:])
                bf_seq_loss=Metrics.compute_BF_seq_loss(logit=bf_y_seq,label=bf_trajectory[1:])
                p_seq_loss=Metrics.compute_predecessor_seq_loss(logit=edge_score_seq,label=p_idx_trajectory[1:],edge_index=batch.edge_index)
                bfs_tau_seq_loss=Metrics.compute_tau_seq_loss(logit=bfs_tau_seq,label=bfs_tau_seq_label)
                bf_tau_seq_loss=Metrics.compute_tau_seq_loss(logit=bf_tau_seq,label=bf_tau_seq_label)
                total_loss=bfs_seq_loss+bf_seq_loss+p_seq_loss+bfs_tau_seq_loss+bf_tau_seq_loss

                """
                wandb
                """
                epoch_bfs_loss.append(bfs_seq_loss)
                epoch_bf_loss.append(bf_seq_loss)
                epoch_p_loss.append(p_seq_loss)
                epoch_bfs_tau_loss.append(bfs_tau_seq_loss)
                epoch_bf_tau_loss.append(bf_tau_seq_loss)
                epoch_total_loss.append(total_loss)

                """
                back propagation
                """
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            """
            Early Stopping
            """
            algo_loss=torch.stack(epoch_bf_loss).mean()
            pre_model=early_stop(val_loss=algo_loss,model=model)
            if early_stop.early_stop:
                model=pre_model
                print(f"Early Stopping in epoch {epoch+1}")
                break

            """
            wandb log
            """
            if config['wandb']:
                wandb.log({
                    'bfs_loss':torch.stack(epoch_bfs_loss).mean(),
                    'bf_loss':torch.stack(epoch_bf_loss).mean(),
                    'p_loss':torch.stack(epoch_p_loss).mean(),
                    'bfs_tau_loss':torch.stack(epoch_bfs_tau_loss).mean(),
                    'bf_tau_loss':torch.stack(epoch_bf_tau_loss).mean(),
                    'total_loss':torch.stack(epoch_total_loss).mean()
                },step=epoch)
            """
            validate
            """
            for val_graph_type,val_data_loader in val_data_loader_dict.items():
                ModelTrainer.test_simultaneously(model=model,graph_type=val_graph_type,data_loader=val_data_loader,config=config)
        return model

    @staticmethod
    def test(model,graph_type,data_loader,config):
        if config['mode']=="test":
            match config['processor'],config['task']:
                case 'mpnn','bfs':
                    model=NGAE_MPNN_BFS(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
                case 'mpnn','bf':
                    model=NGAE_MPNN_BF(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
                case 'gat','bfs':
                    model=NGAE_GAT_BFS(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
                case 'gat','bf':
                    model=NGAE_GAT_BF(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])

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
                
    
    @staticmethod
    def test_simultaneously(model,graph_type,data_loader,config):
        if config['mode']=="test":
            match config['processor']:
                case 'mpnn':
                    model=NGAE_MPNN(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
                case 'gat':
                    model=NGAE_GAT(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
            model=DataUtils.DataLoader.load_model_parameter(model=model,model_name=config['model_name'])
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        bfs_step_acc_list=[]
        bfs_last_acc_list=[]
        p_step_acc_list=[]
        p_last_acc_list=[]
        bfs_tau_step_acc_list=[]
        bfs_tau_last_acc_list=[]
        bf_tau_step_acc_list=[]
        bf_tau_last_acc_list=[]

        with torch.no_grad():
            for batch in tqdm(data_loader,desc=f"{config['mode']} {graph_type} graph..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                bfs_trajectory=batch.bfs
                bf_trajectory=batch.bf 
                p_idx_trajectory=batch.p_idx
                bfs_tau_seq_label=batch.bfs_tau 
                bf_tau_seq_label=batch.bf_tau

                output=model(bfs_trajectory=bfs_trajectory,bf_trajectory=bf_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,mode='train')
                bfs_y_seq=output['bfs_y']
                bf_y_seq=output['bf_y']
                edge_score_seq=output['edge_score']   
                bfs_tau_seq=output['bfs_tau']
                bf_tau_seq=output['bf_tau']

                """
                compute acc
                """
                batch_bfs_step_acc,batch_bfs_last_acc=Metrics.compute_BFS_seq_acc(logit=bfs_y_seq,label=bfs_trajectory[1:])
                batch_p_step_acc,batch_p_last_acc=Metrics.compute_predecessor_seq_acc(logit=edge_score_seq,label=p_idx_trajectory[1:],edge_index=batch.edge_index)
                batch_bfs_tau_step_acc,batch_bfs_tau_last_acc=Metrics.compute_tau_seq_acc(logit=bfs_tau_seq,label=bfs_tau_seq_label)
                batch_bf_tau_step_acc,batch_bf_tau_last_acc=Metrics.compute_tau_seq_acc(logit=bf_tau_seq,label=bf_tau_seq_label)
                bfs_step_acc_list.append(batch_bfs_step_acc)
                bfs_last_acc_list.append(batch_bfs_last_acc)
                p_step_acc_list.append(batch_p_step_acc)
                p_last_acc_list.append(batch_p_last_acc)
                bfs_tau_step_acc_list.append(batch_bfs_tau_step_acc)
                bfs_tau_last_acc_list.append(batch_bfs_tau_last_acc)
                bf_tau_step_acc_list.append(batch_bf_tau_step_acc)
                bf_tau_last_acc_list.append(batch_bf_tau_last_acc)
        """
        print acc
        """
        bfs_step_acc=np.mean(bfs_step_acc_list)
        bfs_last_acc=np.mean(bfs_last_acc_list)
        p_step_acc=np.mean(p_step_acc_list)
        p_last_acc=np.mean(p_last_acc_list)
        bfs_tau_step_acc=np.mean(bfs_tau_step_acc_list)
        bfs_tau_last_acc=np.mean(bfs_tau_last_acc_list)
        bf_tau_step_acc=np.mean(bf_tau_step_acc_list)
        bf_tau_last_acc=np.mean(bf_tau_last_acc_list)

        print(f"{config['mode']} {graph_type} graph BFS step acc: {bfs_step_acc} last acc: {bfs_last_acc}")
        print(f"{config['mode']} {graph_type} graph predecessor step acc: {p_step_acc} last acc: {p_last_acc}")
        print(f"{config['mode']} {graph_type} graph BFS tau step acc: {bfs_tau_step_acc} last acc: {bfs_tau_last_acc}")
        print(f"{config['mode']} {graph_type} graph BF tau step acc: {bf_tau_step_acc} last acc: {bf_tau_last_acc}")