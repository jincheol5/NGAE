import numpy as np
import torch
import torch.nn.functional as F
from .model_train_utils import ModelTrainUtils


class Metrics:
    @staticmethod
    def compute_BFS_seq_loss(logit: torch.Tensor,label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,N,1]
            -label: [seq_len-1,N,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        total_loss=torch.zeros((),device=logit.device) # []
        seq_len,_,_=logit.size()
        for i in range(seq_len):
            total_loss+=F.binary_cross_entropy_with_logits(input=logit[i],target=label[i])
        return total_loss

    @staticmethod
    def compute_BF_seq_loss(logit: torch.Tensor, label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,N,1]
            -label: [seq_len-1,N,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        total_loss=torch.zeros((),device=logit.device) # []
        seq_len,_,_=logit.size()
        for i in range(seq_len):
            total_loss+=F.mse_loss(input=logit[i],target=label[i])
        return total_loss

    @staticmethod
    def compute_predecessor_loss(logit: torch.Tensor,label: torch.Tensor,edge_index: torch.Tensor):
        """
        Input:
            -logit: [E,1]
            -label: [N,1]
            -edge_index: [2,E]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        num_nodes,_=label.size()
        _,dst=edge_index
        total_loss=torch.zeros((),device=logit.device) # []

        logit=logit.view(-1) # [E,1] -> [E,]

        for node_i in range(num_nodes):
            mask=(dst==node_i) # [E,] bool
            incoming_node_score=logit[mask] # [C_i,]
            incoming_node_score=incoming_node_score.unsqueeze(0) # [1,C_i]
            loss_i=F.cross_entropy(input=incoming_node_score,target=label[node_i]) # [1,C_i], [1]
            total_loss+=loss_i
        return total_loss
    
    @staticmethod
    def compute_predecessor_seq_loss(logit: torch.Tensor,label: torch.Tensor,edge_index: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,E,1]
            -label: [seq_len-1,N,1]
            -edge_index: [2,E]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        seq_len,_,_=label.size()
        total_loss=torch.zeros((),device=logit.device) # []
        for i in range(seq_len):
            loss_i=Metrics.compute_predecessor_loss(logit=logit[i],label=label[i],edge_index=edge_index)
            total_loss+=loss_i
        return total_loss

    @staticmethod
    def compute_tau_seq_loss(logit: torch.Tensor,label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,1]
            -label: [seq_len-1,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        return F.binary_cross_entropy_with_logits(input=logit,target=label)

    @staticmethod
    def compute_BFS_acc(logit: torch.Tensor,label: torch.Tensor):
        """
        Input:
            -logit: [N,1]
            -label: [N,1]
        Output:
            -acc
        """
        num_nodes,_=logit.size(0)
        prob=F.sigmoid(logit)
        pred=(prob>=0.5).float() 
        correct=(pred==label).sum().item()
        return float(correct)/num_nodes

    @staticmethod
    def compute_BFS_seq_acc(logit: torch.Tensor,label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,N,1]
            -label: [seq_len-1,N,1]
        Output:
            -step_acc
            -last_acc
        """
        seq_len,_,_=label.size()
        step_acc_list=[]
        for i in range(seq_len):
            step_acc=Metrics.compute_BFS_acc(logit=logit[i],label=label[i])
            step_acc_list.append(step_acc)
        last_acc=step_acc_list[-1]
        step_acc=np.mean(step_acc_list)
        return step_acc,last_acc

    @staticmethod
    def compute_predecessor_acc(logit: torch.Tensor,label: torch.Tensor,edge_index: torch.Tensor):
        """
        Input:
            -logit: [E,1]
            -label: [N,1]
            -edge_index: [2,E]
        Output:
            -acc
        """
        num_nodes,_=label.size()
        p_idx=ModelTrainUtils.compute_predecessor_idx_from_edge_score(edge_score=logit,edge_index=edge_index,num_nodes=num_nodes)
        correct=(p_idx==label).sum().item()
        acc=float(correct)/num_nodes
        return acc
    
    @staticmethod
    def compute_predecessor_seq_acc(logit: torch.Tensor,label: torch.Tensor,edge_index: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,E,1]
            -label: [seq_len-1,N,1]
            -edge_index: [2,E]
        Output:
            -step_acc
            -last_acc
        """
        seq_len,_,_=label.size()
        step_acc_list=[]
        for i in range(seq_len):
            step_acc=Metrics.compute_predecessor_acc(logit=logit[i],label=label[i],edge_index=edge_index)
            step_acc_list.append(step_acc)
        last_acc=step_acc_list[-1]
        step_acc=np.mean(step_acc_list)
        return step_acc,last_acc
    
    @staticmethod
    def compute_tau_seq_acc(logit: torch.Tensor,label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,1]
            -label: [seq_len-1,1]
        Output:
            -step_acc
            -last_acc
        """
        seq_len,_=logit.size()
        prob=F.sigmoid(logit)
        pred=(prob>0.5).float()  
        correct=(pred==label).sum().item()
        step_acc=float(correct)/seq_len
        last_tau=pred[-1,0].item()
        last_acc=1.0 if last_tau==0.0 else 0.0
        return step_acc,last_acc