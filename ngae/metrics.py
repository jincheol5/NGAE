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
        # total_loss=torch.zeros((),device=logit.device) # []
        # seq_len,_,_=logit.size()
        # for i in range(seq_len):
        #     total_loss+=F.binary_cross_entropy_with_logits(input=logit[i],target=label[i])
        # return total_loss
        loss_per_element=F.binary_cross_entropy_with_logits(input=logit,target=label,reduction='mean') # [seq_len-1,N,1]
        loss_per_seq=loss_per_element.mean(dim=(1,2)) # [seq_len,]
        return loss_per_seq.sum()

    @staticmethod
    def compute_BF_seq_loss(logit: torch.Tensor, label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,N,1]
            -label: [seq_len-1,N,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        # total_loss=torch.zeros((),device=logit.device) # []
        # seq_len,_,_=logit.size()
        # for i in range(seq_len):
        #     total_loss+=F.mse_loss(input=logit[i],target=label[i])
        # return total_loss
        loss_per_element=F.mse_loss(input=logit,target=label,reduction='none') # [seq_len-1,N,1]
        loss_per_seq=loss_per_element.mean(dim=(1,2)) # [seq_len,]
        return loss_per_seq.sum()

    @staticmethod
    def compute_predecessor_loss_with_torch_cross_entropy(logit: torch.Tensor,label: torch.Tensor,edge_index: torch.Tensor):
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
    def compute_predecessor_loss(logit: torch.Tensor,label: torch.Tensor,edge_index: torch.Tensor):
        """
        Input:
            -logit: [E,1]
            -label: [N,1]
            -edge_index: [2,E]
        Output:
            -loss scalar tensor: [] (0차원)
        
        Custom cross entropy
        """
        scores=logit.view(-1) # [E,]
        dst=edge_index[1] # [E,]
        num_nodes,_=label.size() 

        # 노드별로 LogSumExp 계산
        m_i=torch.zeros(num_nodes,device=scores.device).scatter_reduce("amax",0,dst,scores) # [N,], 각 노드로 들어오는 edge 중 최대 logit m_i
        exp_norm=torch.exp(scores-m_i[dst]) # [E,]
        sum_exp=torch.zeros_like(m_i).scatter_add_(0,dst,exp_norm) # [N,], 정규화 지수 합
        lse=m_i+torch.log(sum_exp) # [N,], LogSumExp

        # 정답 edge logit 추출, "전체 E개 엣지에서 몇 번째인지" 반환해 올바른 logit 추출
        counts=torch.zeros(num_nodes,dtype=torch.long,device=scores.device).scatter_add_(0,dst,torch.ones_like(dst)) # [N,], 노드별로 들어오는 edge 개수
        offsets=torch.cat([torch.zeros(1,dtype=torch.long,device=scores.device),counts.cumsum(0)[:-1]],dim=0) # [N,], offsets=node i 이전에 등장한(index가 작은) 모든 edge 개수 누적

        label_abs=offsets+label.view(-1) # [N,]
        true_scores=scores[label_abs] # [N,]
        return (lse-true_scores).sum()

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
        num_nodes,_=logit.size()
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