import numpy as np
import torch
import torch.nn.functional as F
from .model_train_utils import ModelTrainUtils


class Metrics:
    @staticmethod
    def compute_r_seq_loss(logit: torch.Tensor,label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,N,1]
            -label: [seq_len-1,N,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        return F.binary_cross_entropy_with_logits(input=logit,target=label,reduction='mean')

    @staticmethod
    def compute_d_seq_loss(logit: torch.Tensor, label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,N,1]
            -label: [seq_len-1,N,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        return F.mse_loss(input=logit,target=label,reduction='mean')

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
        m_i=torch.zeros(num_nodes,device=scores.device).scatter_reduce(dim=0,index=dst,src=scores,reduce="amax") # [N,], 각 노드로 들어오는 edge 중 최대 logit m_i
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
        losses=[Metrics.compute_predecessor_loss(logit=logit[i],label=label[i],edge_index=edge_index) for i in range(seq_len)]
        return torch.stack(losses).mean()

    @staticmethod
    def compute_tau_seq_loss(logit: torch.Tensor,label: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,1]
            -label: [seq_len-1,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        return F.binary_cross_entropy_with_logits(input=logit,target=label,reduction='mean')

    @staticmethod
    def compute_r_acc(logit: torch.Tensor,label: torch.Tensor):
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
    def compute_r_seq_acc(logit: torch.Tensor,label: torch.Tensor):
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
            step_acc=Metrics.compute_r_acc(logit=logit[i],label=label[i])
            step_acc_list.append(step_acc)
        last_acc=step_acc_list[-1]
        step_acc=np.mean(step_acc_list)
        return step_acc,last_acc

    @staticmethod
    def compute_predecessor_acc(logit: torch.Tensor,label: torch.Tensor,edge_index: torch.Tensor):
        """
        Input:
            -logit: [E,1] edge_score tensor
            -label: [N,1] p tensor
        """
        num_nodes=label.size(0)
        softmax_one_hot_p=ModelTrainUtils.convert_edge_score_to_softmax_one_hot_p(edge_score=logit,edge_index=edge_index,num_nodes=num_nodes) # [N,N,1]
        softmax_one_hot_p=softmax_one_hot_p.squeeze(-1) # [N,N]
        pred_p=softmax_one_hot_p.argmax(dim=1,keepdim=True) # [N,1], LongTensor
        correct=(pred_p==label).sum().item()
        acc=float(correct)/num_nodes
        return acc

    @staticmethod
    def compute_predecessor_seq_acc(logit: torch.Tensor,label: torch.Tensor,edge_index: torch.Tensor):
        """
        Input:
            -logit: [seq_len-1,E,1] edge_score seq tensor
            -label: [seq_len-1,N,1] p seq tensor
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