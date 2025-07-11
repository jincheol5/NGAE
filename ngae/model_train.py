import numpy as np
import torch
from tqdm import tqdm
from .metrics import Metrics


class ModelTrainer:
    @staticmethod
    def train(model,train_data_loader,val_data_loader_dict,config: dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])

        for epoch in tqdm(range(config['epochs']),desc=f"Training {config['task']}..."):
            model.train()
            for batch in tqdm(train_data_loader,desc=f"Epoch {epoch}..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                algo_trajectory=batch.bf # [seq_len,N,1]
                p_idx_trajectory=batch.p_idx # [seq_len,N,1]
                tau_seq_label=batch.tau # [seq_len-1,1]

                output=model(algo_trajectory=algo_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,task='train')
                y_seq=output['y'] # [seq_len-1,N,1]
                edge_score_seq=output['edge_score'] # [seq_len-1,E,1]
                tau_seq=output['tau'] # [seq_len-1,1]

                total_loss=torch.zeros((),device=device)
                y_seq_loss=Metrics.compute_BF_seq_loss(logit=y_seq,label=algo_trajectory[1:])
                edge_score_seq_loss=Metrics.compute_predecessor_seq_loss(logit=edge_score_seq,label=p_idx_trajectory[1:],edge_index=batch.edge_index)
                tau_seq_loss=Metrics.compute_tau_seq_loss(logit=tau_seq,label=tau_seq_label)
                total_loss=y_seq_loss+edge_score_seq_loss+tau_seq_loss

                """
                back propagation
                """
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            """
            validate
            """
            for val_graph_type,val_data_loader in val_data_loader_dict.items():
                ModelTrainer.validate(model=model,val_graph_type=val_graph_type,val_data_loader=val_data_loader,config=config)

    @staticmethod
    def validate(model,val_graph_type,val_data_loader,config):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        p_step_acc_list=[]
        p_last_acc_list=[]
        tau_step_acc_list=[]
        tau_last_acc_list=[]
        with torch.no_grad():
            for batch in tqdm(val_data_loader,desc=f"Validate {val_graph_type} graph..."):
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
                h_0=h_0.to(device)

                algo_trajectory=batch.bf # [seq_len,N,1]
                p_idx_trajectory=batch.p_idx # [seq_len,N,1]
                tau_seq_label=batch.tau # [seq_len-1,1]

                output=model(algo_trajectory=algo_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,task='test')
                y_seq=output['y'] # [seq_len-1,N,1]
                edge_score_seq=output['edge_score'] # [seq_len-1,E,1]
                tau_seq=output['tau'] # [seq_len-1,1]

                """
                compute acc
                """
                batch_p_step_acc,batch_p_last_acc=Metrics.compute_predecessor_seq_acc(logit=edge_score_seq,label=p_idx_trajectory[1:],edge_index=batch.edge_index)
                batch_tau_step_acc,batch_tau_last_acc=Metrics.compute_tau_seq_acc(logit=tau_seq,label=tau_seq_label)
                p_step_acc_list.append(batch_p_step_acc)
                p_last_acc_list.append(batch_p_last_acc)
                tau_step_acc_list.append(batch_tau_step_acc)
                tau_last_acc_list.append(batch_tau_last_acc)
        
        p_step_acc=np.mean(p_step_acc_list)
        p_last_acc=np.mean(p_last_acc_list)
        tau_step_acc=np.mean(tau_step_acc_list)
        tau_last_acc=np.mean(tau_last_acc_list)

        print(f"Validate {val_graph_type} graph task predecessor step acc: {p_step_acc} last acc: {p_last_acc}")
        print(f"Validate {val_graph_type} graph task tau step acc: {tau_step_acc} last acc: {tau_last_acc}")