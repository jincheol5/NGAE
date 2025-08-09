import numpy as np
import wandb
import torch
from tqdm import tqdm
from .metrics import Metrics
from .model import NGAE
from .model_train_utils import EarlyStopping
from .data_utils import DataUtils

class ModelTrainer:
    @staticmethod
    def train(model,train_data_loader,val_data_loader,config:dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])
        early_stop=EarlyStopping(patience=config['patience'])

        for epoch in tqdm(range(config['epochs']),desc=f"Training {config['algorithm']}..."):
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

                match config['algorithm']:
                    case 'bfs':
                        algo_trajectory=batch.r # [seq_len,N,1]
                        tau_seq_label=batch.bfs_tau # [seq_len-1,1]
                    case 'bf':
                        algo_trajectory=batch.d # [seq_len,N,1]
                        p_idx_trajectory=batch.bf_p_idx # [seq_len,N,1]
                        tau_seq_label=batch.bf_tau # [seq_len-1,1]

                output=model(algo_trajectory=algo_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,mode='train')
                algo_seq=output['algo'] # [seq_len-1,N,1]
                tau_seq=output['tau'] # [seq_len-1,1]
                if config['algorithm']=='bf':
                    edge_score_seq=output['edge_score'] # [seq_len-1,E,1]

                match config['algorithm']:
                    case 'bfs':
                        algo_seq_loss=Metrics.compute_r_seq_loss(logit=algo_seq,label=algo_trajectory[1:])
                        tau_seq_loss=Metrics.compute_tau_seq_loss(logit=tau_seq,label=tau_seq_label)
                        total_loss=algo_seq_loss+tau_seq_loss
                    case 'bf':
                        algo_seq_loss=Metrics.compute_d_seq_loss(logit=algo_seq,label=algo_trajectory[1:])
                        p_seq_loss=Metrics.compute_predecessor_seq_loss(logit=edge_score_seq,label=p_idx_trajectory[1:],edge_index=batch.edge_index)
                        tau_seq_loss=Metrics.compute_tau_seq_loss(logit=tau_seq,label=tau_seq_label)
                        total_loss=algo_seq_loss+p_seq_loss+tau_seq_loss

                # wandb
                epoch_algo_loss.append(algo_seq_loss)
                epoch_tau_loss.append(tau_seq_loss)
                epoch_total_loss.append(total_loss)
                if config['algorithm']=='bf':
                    epoch_p_loss.append(p_seq_loss)
                
                """
                back propagation
                """
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            """
            Early stopping
            """
            if config['algorithm']=='bfs':
                val_loss=torch.stack(epoch_algo_loss).mean().item()
            else: # bf
                val_loss=torch.stack(epoch_p_loss).mean().item()
            pre_model=early_stop(val_loss=val_loss,model=model)
            if early_stop.early_stop:
                model=pre_model
                print(f"Early Stopping in epoch {epoch+1}")
                break

            """
            wandb log
            """
            if config['wandb']:
                if config['algorithm']=='bfs':
                    wandb.log({
                        f"r_loss":torch.stack(epoch_algo_loss).mean().item(),
                        f"tau_loss":torch.stack(epoch_tau_loss).mean().item(),
                        f"total_loss":torch.stack(epoch_total_loss).mean().item()
                    },step=epoch)
                else: # bf
                    wandb.log({
                        f"d_loss":torch.stack(epoch_algo_loss).mean().item(),
                        f"p_loss":torch.stack(epoch_p_loss).mean().item(),
                        f"tau_loss":torch.stack(epoch_tau_loss).mean().item(),
                        f"total_loss":torch.stack(epoch_total_loss).mean().item()
                    },step=epoch)
            """
            validate
            """
            ModelTrainer.test(model=model,graph_type='all',data_loader=val_data_loader,config=config)
        return model

    @staticmethod
    def test(model,graph_type,data_loader,config):
        if config['mode']=='test':
            model=NGAE(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'],algorithm=config['algorithm'],processor=config['processor'],aggr=config['aggr'])
            model=DataUtils.DataLoader.load_model_parameter(model=model,model_name=config['model_name'])
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        if config['algorithm']=='bfs':
            r_step_acc_list=[]
            r_last_acc_list=[]
        else: # bf
            p_step_acc_list=[]
            p_last_acc_list=[]
        tau_step_acc_list=[]
        tau_last_acc_list=[]

        with torch.no_grad():
            for batch in tqdm(data_loader,desc=f"{config['mode']} {graph_type} graph..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                match config['algorithm']:
                    case 'bfs':
                        algo_trajectory=batch.r # [seq_len,N,1]
                        tau_seq_label=batch.bfs_tau # [seq_len-1,1]
                    case 'bf':
                        algo_trajectory=batch.d # [seq_len,N,1]
                        p_trajectory=batch.bf_p # [seq_len,N,1]
                        tau_seq_label=batch.bf_tau # [seq_len-1,1]
                
                output=model(algo_trajectory=algo_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,mode='test')
                algo_seq=output['algo'] # [seq_len-1,N,1]
                tau_seq=output['tau'] # [seq_len-1,1]
                if config['algorithm']=='bf':
                    edge_score_seq=output['edge_score'] # [seq_len-1,E,1]
                
                """
                compute acc
                """
                if config['algorithm']=='bfs':
                    batch_r_step_acc,batch_r_last_acc=Metrics.compute_r_seq_acc(logit=algo_seq,label=algo_trajectory[1:])
                    r_step_acc_list.append(batch_r_step_acc)
                    r_last_acc_list.append(batch_r_last_acc)
                else: # bf
                    batch_p_step_acc,batch_p_last_acc=Metrics.compute_predecessor_seq_acc(logit=edge_score_seq,label=p_trajectory[1:],edge_index=batch.edge_index)
                    p_step_acc_list.append(batch_p_step_acc)
                    p_last_acc_list.append(batch_p_last_acc)
                batch_tau_step_acc,batch_tau_last_acc=Metrics.compute_tau_seq_acc(logit=tau_seq,label=tau_seq_label)
                tau_step_acc_list.append(batch_tau_step_acc)
                tau_last_acc_list.append(batch_tau_last_acc)

        match config['algorithm']:
            case 'bfs':
                r_step_acc=np.mean(r_step_acc_list)
                r_last_acc=np.mean(r_last_acc_list)
                tau_step_acc=np.mean(tau_step_acc_list)
                tau_last_acc=np.mean(tau_last_acc_list)

                print(f"{config['mode']} {graph_type} graph r step acc: {r_step_acc} last acc: {r_last_acc}")
                print(f"{config['mode']} {graph_type} graph tau step acc: {tau_step_acc} last acc: {tau_last_acc}")
            case 'bf':
                p_step_acc=np.mean(p_step_acc_list)
                p_last_acc=np.mean(p_last_acc_list)
                tau_step_acc=np.mean(tau_step_acc_list)
                tau_last_acc=np.mean(tau_last_acc_list)

                print(f"{config['mode']} {graph_type} graph predecessor step acc: {p_step_acc} last acc: {p_last_acc}")
                print(f"{config['mode']} {graph_type} graph tau step acc: {tau_step_acc} last acc: {tau_last_acc}")