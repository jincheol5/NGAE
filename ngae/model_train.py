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

        """
        Early stopping
        """
        if config['early_stop']:
            early_stop=EarlyStopping(patience=config['patience'])

        """
        model train
        """
        for epoch in tqdm(range(config['epochs']),desc=f"Training {config['algorithm']}..."):
            # wandb
            epoch_algo_loss=[]
            epoch_p_loss=[]
            epoch_tau_loss=[]
            epoch_total_loss=[]

            model.train()
            for batch in tqdm(train_data_loader,desc=f"Epoch {epoch+1}..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                match config['algorithm']:
                    case 'bfs':
                        algo_trajectory=batch.r.permute(1,0,2).contiguous()
                        algo_mask_trajectory=batch.bfs_mask.permute(1,0,2).contiguous()
                        algo_mask_trajectory=torch.cat([algo_mask_trajectory[1:],algo_mask_trajectory[-1:]],dim=0)
                        tau_label=batch.bfs_tau.permute(1,0).contiguous()
                        tau_mask_trajectory=batch.bfs_tau_mask.permute(1,0).contiguous()
                    case 'bf':
                        algo_trajectory=batch.d.permute(1,0,2).contiguous()
                        algo_mask_trajectory=batch.bf_mask.permute(1,0,2).contiguous()
                        algo_mask_trajectory=torch.cat([algo_mask_trajectory[1:],algo_mask_trajectory[-1:]],dim=0)
                        p_trajectory=batch.bf_p.permute(1,0,2).contiguous()
                        tau_label=batch.bf_tau.permute(1,0).contiguous()
                        tau_mask_trajectory=batch.bf_tau_mask.permute(1,0).contiguous()

                output=model(
                    algo_trajectory=algo_trajectory,
                    algo_mask_trajectory=algo_mask_trajectory,
                    h_0=h_0,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    algo_type=config['algorithm'],
                    mode='train')
                
                pred_algo=output['algo'] # List of [sub_B*N,1]
                pred_edge_score=output['edge_score'] # List of [sub_E,1]
                pred_tau=output['tau'] # List of [sub_B,]
                sub_edge_index_list=output['sub_edge_index'] # List of sub_edge_index


                num_nodes=batch.num_nodes//batch.num_graphs
                algo_loss=Metrics.compute_algo_seq_loss(logit=pred_algo,label=algo_trajectory[1:],mask=algo_mask_trajectory[:-1],num_nodes=num_nodes,algo_type=config['algorithm'])
                tau_loss=Metrics.compute_tau_seq_loss(logit=pred_tau,label=tau_label,mask=tau_mask_trajectory)
                if config['algorithm']=='bfs':
                    total_loss=algo_loss+tau_loss
                else: # bf
                    p_loss=Metrics.compute_p_seq_loss(logit=pred_edge_score,label=p_trajectory[1:],mask=algo_mask_trajectory[:-1],edge_index_list=sub_edge_index_list,num_nodes=num_nodes)
                    total_loss=algo_loss+p_loss+tau_loss

                # wandb
                epoch_algo_loss.append(algo_loss)
                epoch_tau_loss.append(tau_loss)
                epoch_total_loss.append(total_loss)
                if config['algorithm']=='bf':
                    epoch_p_loss.append(p_loss)

                # back propagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            """
            Early stopping
            """
            if config['early_stop']:
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
                        f"{config['algorithm']}_loss":torch.stack(epoch_algo_loss).mean().item(),
                        f"tau_loss":torch.stack(epoch_tau_loss).mean().item(),
                        f"total_loss":torch.stack(epoch_total_loss).mean().item()
                    },step=epoch)
                else: # bf
                    wandb.log({
                        f"{config['algorithm']}_loss":torch.stack(epoch_algo_loss).mean().item(),
                        f"{config['algorithm']}_p_loss":torch.stack(epoch_p_loss).mean().item(),
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
            model=NGAE(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'],processor=config['processor'],aggr=config['aggr'])
            model=DataUtils.DataLoader.load_model_parameter(model=model,model_name=config['model_name'])
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        algo_step_acc_list=[]
        algo_last_acc_list=[]
        p_step_acc_list=[]
        p_last_acc_list=[]
        tau_step_acc_list=[]
        tau_last_acc_list=[]

        """
        model test
        """
        with torch.no_grad():
            for batch in tqdm(data_loader,desc=f"{config['mode']} {graph_type} graph..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                match config['algorithm']:
                    case 'bfs':
                        algo_trajectory=batch.r.permute(1,0,2).contiguous()
                        algo_mask_trajectory=batch.bfs_mask.permute(1,0,2).contiguous()
                        algo_mask_trajectory=torch.cat([algo_mask_trajectory[1:],algo_mask_trajectory[-1:]],dim=0)
                        tau_label=batch.bfs_tau.permute(1,0).contiguous()
                        tau_mask_trajectory=batch.bfs_tau_mask.permute(1,0).contiguous()
                    case 'bf':
                        algo_trajectory=batch.d.permute(1,0,2).contiguous()
                        algo_mask_trajectory=batch.bf_mask.permute(1,0,2).contiguous()
                        algo_mask_trajectory=torch.cat([algo_mask_trajectory[1:],algo_mask_trajectory[-1:]],dim=0)
                        p_trajectory=batch.bf_p.permute(1,0,2).contiguous()
                        tau_label=batch.bf_tau.permute(1,0).contiguous()
                        tau_mask_trajectory=batch.bf_tau_mask.permute(1,0).contiguous()

                output=model(
                    algo_trajectory=algo_trajectory,
                    algo_mask_trajectory=algo_mask_trajectory,
                    h_0=h_0,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    algo_type=config['algorithm'],
                    mode='test')
                
                pred_algo=output['algo'] # List of [sub_B*N,1]
                pred_edge_score=output['edge_score'] # List of [sub_E,1]
                pred_tau=output['tau'] # List of [sub_B,]
                sub_edge_index_list=output['sub_edge_index'] # List of sub_edge_index

                """
                compute batch acc
                """
                num_nodes=batch.num_nodes//batch.num_graphs
                if config['algorithm']=='bfs':
                    batch_algo_step_acc,batch_algo_last_acc=Metrics.compute_r_seq_acc(logit=pred_algo,label=algo_trajectory[1:],mask=algo_mask_trajectory[:-1],num_nodes=num_nodes)
                    algo_step_acc_list.append(batch_algo_step_acc)
                    algo_last_acc_list.append(batch_algo_last_acc)
                else: # bf
                    batch_p_step_acc,batch_p_last_acc=Metrics.compute_p_seq_acc(logit=pred_edge_score,label=p_trajectory[1:],mask=algo_mask_trajectory[:-1],edge_index_list=sub_edge_index_list,num_nodes=num_nodes)
                    p_step_acc_list.append(batch_p_step_acc)
                    p_last_acc_list.append(batch_p_last_acc)
                batch_tau_step_acc,batch_tau_last_acc=Metrics.compute_tau_seq_acc(logit=pred_tau,label=tau_label,mask=tau_mask_trajectory)
                tau_step_acc_list.append(batch_tau_step_acc)
                tau_last_acc_list.append(batch_tau_last_acc)
        
        """
        compute acc
        """
        if config['algorithm']=='bfs':
            algo_step_acc=np.mean(algo_step_acc_list)
            algo_last_acc=np.mean(algo_last_acc_list)
            print(f"{config['mode']} {graph_type} graph {config['algorithm']} step acc: {algo_step_acc} last acc: {algo_last_acc}")
        else: # bf
            p_step_acc=np.mean(p_step_acc_list)
            p_last_acc=np.mean(p_last_acc_list)
            print(f"{config['mode']} {graph_type} graph {config['algorithm']} predecessor step acc: {p_step_acc} last acc: {p_last_acc}")
        tau_step_acc=np.mean(tau_step_acc_list)
        tau_last_acc=np.mean(tau_last_acc_list)
        print(f"{config['mode']} {graph_type} graph tau step acc: {tau_step_acc} last acc: {tau_last_acc}")