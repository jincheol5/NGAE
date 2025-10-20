import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from typing_extensions import Literal
from torcheval.metrics import BinaryAUROC
from .model_train_utils import ModelTrainUtils

class Metrics:
    @staticmethod
    def compute_algo_step_loss(logit:torch.Tensor, label:torch.Tensor, mask:torch.Tensor, num_nodes:int, algo_type:Literal['bfs','bf']='bfs'):
        """
        Input:
            logit: [sub_B*N,1]
            label: [B*N,1]
            mask: [B*N,]
            num_nodes: int
            algo_type: bfs or bf
        Output:
            loss scalar tensor: [] (0차원)
        """
        sub_batch_size=logit.size(0)//num_nodes
        logit=logit.view(sub_batch_size,num_nodes,1).squeeze(-1) # [sub_B,N]
        label=label[mask] # [sub_B*N,1]
        label=label.view(sub_batch_size,num_nodes,1).squeeze(-1) # [sub_B,N]
        match algo_type:
            case 'bfs':
                loss_per_subgraph=F.binary_cross_entropy_with_logits(logit,label,reduction='none').mean(dim=1) # [sub_B,]
            case 'bf':
                loss_per_subgraph=F.mse_loss(logit,label,reduction='none').mean(dim=1) # [sub_B,]
        loss=loss_per_subgraph.mean()
        return loss

    @staticmethod
    def compute_algo_seq_loss(logit:list, label:torch.Tensor, mask:torch.Tensor, num_nodes:int, algo_type:Literal['bfs','bf']='bfs'):
        """
        Input:
            logit: List of [sub_B*N,1]
            label: [max_seq_len-1,B*N,1]
            mask: [max_seq_len-1,B*N,1]
            num_nodes: int 
            algo_type: bfs or bf
        Output:
            loss scalar tensor: [] (0차원)
        """
        seq_len=label.size(0)
        losses=[]
        for step in range(seq_len):
            step_logit=logit[step] # [sub_B*N,1]
            step_label=label[step] # [B*N,1]
            step_mask=mask[step].squeeze(-1) # [B*N,]
            step_loss=Metrics.compute_algo_step_loss(logit=step_logit,label=step_label,mask=step_mask,num_nodes=num_nodes,algo_type=algo_type)
            losses.append(step_loss)
        return torch.stack(losses).mean()

    @staticmethod
    def compute_p_step_loss(logit:torch.Tensor, label:torch.Tensor, mask:torch.Tensor, edge_index:torch.Tensor, num_nodes:int):
        """
        Input:
            logit (edge_score): [sub_E,1]
            label: [B*N,1]
            mask: [B*N,]
            edge_index: [2,sub_E]
            num_nodes: int
        Output:
            loss scalar tensor: [] (0차원)
        """
        label=label[mask] # [sub_B*N,1]
        updated_label=ModelTrainUtils.offset_p(p=label,num_nodes=num_nodes).squeeze(-1) # [sub_B*N,]

        src,tar=edge_index
        logit=logit.squeeze(-1)
        log_denom=torch_scatter.scatter_logsumexp(src=logit,index=tar,dim=0,dim_size=updated_label.size(0)) # [sub_B*N,]
        log_probs=logit-log_denom[tar] # [sub_E,]
        correct_edge_mask=(src==updated_label[tar])
        loss=-log_probs[correct_edge_mask].mean()
        return loss


    @staticmethod
    def compute_p_seq_loss(logit:list, label:torch.Tensor, mask:torch.Tensor, edge_index_list:list, num_nodes:int):
        """
        Input:
            logit: List of [sub_E,1]
            label: [max_seq_len-1,B*N,1]
            mask: [max_seq_len-1,B*N,1]
            edge_index_list: List of sub_edge_index
            num_nodes: int 
        Output:
            loss scalar tensor: [] (0차원)
        """
        seq_len=label.size(0)
        losses=[]
        for step in range(seq_len):
            step_logit=logit[step] # [sub_E,1]
            step_label=label[step] # [B*N,1]
            step_mask=mask[step].squeeze(-1) # [B*N,]
            step_loss=Metrics.compute_p_step_loss(logit=step_logit,label=step_label,mask=step_mask,edge_index=edge_index_list[step],num_nodes=num_nodes)
            losses.append(step_loss)
        return torch.stack(losses).mean()
    
    @staticmethod
    def compute_tau_step_loss(logit:torch.Tensor, label:torch.Tensor, mask:torch.Tensor):
        """
        Input:
            logit: [sub_B,]
            label: [B,]
            mask: [B,]
        Output:
            loss scalar tensor: [] (0차원)
        """
        label=label[mask] # [sub_B,]
        return F.binary_cross_entropy_with_logits(input=logit,target=label,reduction='mean')

    @staticmethod
    def compute_tau_seq_loss(logit:list ,label:torch.Tensor, mask:torch.Tensor):
        """
        Input:
            logit: List of [sub_B,]
            label: [max_seq_len-1,B]
            mask: [max_seq_len-1,B]
        Output:
            loss scalar tensor: [] (0차원)
        """
        seq_len=label.size(0)
        losses=[]
        for step in range(seq_len):
            step_logit=logit[step] # [sub_B,]
            step_label=label[step] # [B,]
            step_mask=mask[step] # [B,]
            step_loss=Metrics.compute_tau_step_loss(logit=step_logit,label=step_label,mask=step_mask)
            losses.append(step_loss)
        return torch.stack(losses).mean()

    @staticmethod
    def compute_r_step_acc(logit:torch.Tensor, label:torch.Tensor, mask:torch.Tensor, num_nodes:int):
        """
        Input:
            logit: [sub_B*N,1]
            label: [B*N,1]
            mask: [B*N,]
            num_nodes: int
        Output:
            acc
        """
        label=label[mask] # [sub_B*N,1]
        prob=F.sigmoid(logit) # [sub_B*N,1]
        pred=(prob>=0.5).float() # [sub_B*N,1]
        correct=(pred==label).float() # [sub_B*N,1]
        correct=correct.view(-1,num_nodes,1) # [sub_B,N,1]
        acc_per_subgraph=correct.mean(dim=1) # [sub_B,1]
        acc=acc_per_subgraph.mean().item()
        return acc 

    @staticmethod
    def compute_r_seq_acc(logit:list, label:torch.Tensor, mask:torch.Tensor, num_nodes:int):
        """
        Input:
            logit: List of [sub_B*N,1]
            label: [max_seq_len-1,B*N,1]
            mask: [max_seq_len-1,B*N,1]
            num_nodes: int
        Output:
            step_acc
            last_acc
        """
        seq_len=label.size(0)
        acc_list=[]
        for step in range(seq_len):
            step_logit=logit[step] # [sub_B,]
            step_label=label[step] # [B,]
            step_mask=mask[step].squeeze(-1) # [B,]
            step_acc=Metrics.compute_r_step_acc(logit=step_logit,label=step_label,mask=step_mask,num_nodes=num_nodes)
            acc_list.append(step_acc)
        step_acc=np.mean(acc_list)
        last_acc=acc_list[-1]
        return step_acc,last_acc

    @staticmethod
    def compute_p_step_acc(logit:torch.Tensor, label:torch.Tensor, mask:torch.Tensor, edge_index:torch.Tensor, num_nodes:int):
        """
        Input:
            logit (edge_score): [sub_E,1]
            label: [B*N,1]
            mask: [B*N,]
            edge_index: [2,sub_E]
            num_nodes: int
        Output:
            acc
        """
        label=label[mask] # [sub_B*N,1]
        updated_label=ModelTrainUtils.offset_p(p=label,num_nodes=num_nodes) # [sub_B*N,1]
        pred=ModelTrainUtils.compute_p_from_logit(logit=logit,edge_index=edge_index,num_nodes=updated_label.size(0)) # [sub_B*N,1] 
        correct=(pred==updated_label).float() # [sub_B*N,1]
        correct=correct.view(-1,num_nodes,1) # [sub_B,N,1]
        acc_per_subgraph=correct.mean(dim=1) # [sub_B,1]
        acc=acc_per_subgraph.mean().item()
        return acc

    @staticmethod
    def compute_p_seq_acc(logit:list, label:torch.Tensor, mask:torch.Tensor, edge_index_list:torch.Tensor, num_nodes:int):
        """
        Input:
            logit: List of [B*N,1]
            label: [max_seq_len-1,B*N,1]
            mask: [max_seq_len-1,B*N,1]
            edge_index_list: List of sub_edge_index
            num_nodes: int
        Output:
            step_acc
            last_acc
        """
        seq_len=label.size(0)
        acc_list=[]
        for step in range(seq_len):
            step_logit=logit[step] # [sub_B,]
            step_label=label[step] # [B,]
            step_mask=mask[step].squeeze(-1) # [B,]
            step_acc=Metrics.compute_p_step_acc(logit=step_logit,label=step_label,mask=step_mask,edge_index=edge_index_list[step],num_nodes=num_nodes)
            acc_list.append(step_acc)
        step_acc=np.mean(acc_list)
        last_acc=acc_list[-1]
        return step_acc,last_acc

    @staticmethod
    def compute_tau_seq_acc(logit:list, label:torch.Tensor, mask:torch.Tensor):
        """
        Input:
            logit: List of [sub_B]
            label: [max_seq_len-1,B]
            mask: [max_seq_len-1,B]
        Output:
            step_acc
            last_acc
        """
        seq_len=label.size(0)
        acc_list=[]
        for step in range(seq_len):
            step_label=label[step] # [B,]
            step_mask=mask[step] # [B,]
            prob=torch.sigmoid(logit[step]) # [sub_B,] 
            pred=(prob>=0.5).float() # [sub_B,]
            step_label=step_label[step_mask] # [sub_B,]
            step_acc=(pred==step_label).float().mean().item()
            acc_list.append(step_acc)
        step_acc=np.mean(acc_list)
        last_acc=acc_list[-1]
        return step_acc,last_acc