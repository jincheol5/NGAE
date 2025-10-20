import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data,Batch

class ModelTrainUtils:
    @staticmethod
    def batch_collate_fn(batch_data_list:list):
        r_list=[data.r for data in batch_data_list]
        bfs_tau_list=[data.bfs_tau for data in batch_data_list]

        d_list=[data.d for data in batch_data_list]
        bf_p_list=[data.bf_p for data in batch_data_list]
        bf_tau_list=[data.bf_tau for data in batch_data_list]

        padding_value=-1.0
        padded_r=pad_sequence(sequences=r_list,batch_first=True,padding_value=padding_value) # [batch_size,max_seq_len,N,1]
        padded_bfs_tau=pad_sequence(sequences=bfs_tau_list,batch_first=True,padding_value=padding_value) # [batch_size,max_seq_len-1,1]
        padded_d=pad_sequence(sequences=d_list,batch_first=True,padding_value=padding_value) # [batch_size,max_seq_len,N,1]
        padded_bf_p=pad_sequence(sequences=bf_p_list,batch_first=True,padding_value=padding_value) # [batch_size,max_seq_len,N,1]
        padded_bf_tau=pad_sequence(sequences=bf_tau_list,batch_first=True,padding_value=padding_value) # [batch_size,max_seq_len-1,1]

        padded_r_list=[padded_r[b] for b in range(padded_r.size(0))] # list of [max_seq_len,N,1]
        padded_bfs_tau_list=[padded_bfs_tau[b] for b in range(padded_bfs_tau.size(0))] # list of [max_seq_len-1,1]
        padded_d_list=[padded_d[b] for b in range(padded_d.size(0))] # list of [max_seq_len,N,1]
        padded_bf_p_list=[padded_bf_p[b] for b in range(padded_bf_p.size(0))] # list of [max_seq_len,N,1]
        padded_bf_tau_list=[padded_bf_tau[b] for b in range(padded_bf_tau.size(0))] # list of [max_seq_len-1,1]

        updated_batch_data_list=[]
        _,num_nodes,_=r_list[0].size()
        for idx,data in enumerate(batch_data_list):
            updated_data=Data(edge_index=data.edge_index,edge_attr=data.edge_attr,num_nodes=num_nodes)

            updated_data.r=padded_r_list[idx].permute(1,0,2).contiguous() # [max_seq_len,N,1]->[N,max_seq_len,1]
            updated_data.bfs_tau=padded_bfs_tau_list[idx].permute(1,0).contiguous() # [max_seq_len-1,1]->[1,max_seq_len-1]

            updated_data.d=padded_d_list[idx].permute(1,0,2).contiguous() # [max_seq_len,N,1]->[N,max_seq_len,1]
            updated_data.bf_p=padded_bf_p_list[idx].permute(1,0,2).contiguous() # [max_seq_len,N,1]->[N,max_seq_len,1]
            updated_data.bf_tau=padded_bf_tau_list[idx].permute(1,0).contiguous() # [max_seq_len-1,1]->[1,max_seq_len-1]

            updated_data.bfs_mask=(updated_data.r!=padding_value).contiguous() # [N,max_seq_len,1], mask of r,bfs_p
            updated_data.bfs_tau_mask=(updated_data.bfs_tau!=padding_value).contiguous() # [1,max_seq_len-1], mask of bfs_tau

            updated_data.bf_mask=(updated_data.d!=padding_value).contiguous() # [N,max_seq_len,1], mask of d,bf_p
            updated_data.bf_tau_mask=(updated_data.bf_tau!=padding_value).contiguous() # [1,max_seq_len-1], mask of bf_tau

            updated_batch_data_list.append(updated_data)
        return Batch.from_data_list(updated_batch_data_list)

    @staticmethod
    def offset_p(p:torch.Tensor, num_nodes:int):
        """
        Input:
            p: [N,1], long tensor
            num_nodes: int
        Output:
            offset_p: [N,1], long tensor
        """
        sub_len=p.size(0)//num_nodes
        for i in range(sub_len):
            offset_value=num_nodes*i
            p[num_nodes*i:num_nodes*(i+1)]=p[num_nodes*i:num_nodes*(i+1)]+offset_value
        return p 
    
    @staticmethod
    def re_offset_p(p:torch.Tensor, num_nodes:int, mask:torch.Tensor):
        """
        Input:
            p: [sub_N,1], long tensor (offset된 상태)
            num_nodes: int
        Output:
            restored_p: [N,1], long tensor (offset 이전 상태)
        """
        sub_len=p.size(0)//num_nodes
        for i in range(sub_len):
            offset_value=num_nodes*i
            p[num_nodes*i:num_nodes*(i+1)]=p[num_nodes*i:num_nodes*(i+1)]-offset_value
        return p

    @staticmethod
    def convert_p_to_edge_p(p:torch.Tensor, edge_index:torch.Tensor):
        """
        Input:
            p: [N,1], long tensor
            edge_index: [2,E]
        Output:
            edge_p: [E,1]
        """
        p=p.squeeze(-1)
        src,tar=edge_index
        edge_p=(p[tar]==src).float().unsqueeze(1)
        return edge_p # [E,1]

    @staticmethod
    def restore_output(output:torch.Tensor, mask:torch.Tensor, feature_dim:int=1):
        """
        Input:
            output: [sub_N,feature_dim]
            mask: [N,]
            feature_dim: int
        Output:
            restored_output: [N,feature_dim]
        """
        restored_output=torch.full((mask.size(0),feature_dim),-1.0,dtype=output.dtype,device=output.device)
        restored_output[mask]=output
        return restored_output

    @staticmethod
    def compute_r_from_logit(logit:torch.Tensor):
        """
        Input:
            logit: [N,1]
        Output:
            r: [N,1]
        """
        prob=F.sigmoid(logit)           
        mask=prob>=0.5                   
        return mask.to(logit.dtype)

    @staticmethod
    def compute_p_from_logit(logit:torch.Tensor, edge_index:torch.Tensor, num_nodes:int):
        """
        Input:
            logit (edge_score): [E,1]
            edge_index: [2,E]
            num_nodes: int
        Output:
            p: [N,1]
        """
        src,tar=edge_index
        logit=logit.squeeze(-1)
        max_scores,argmax_indices=torch_scatter.scatter_max(logit,tar,dim=0,dim_size=num_nodes)
        p=src[argmax_indices].unsqueeze(-1)
        return p

    @staticmethod
    def teacher_forcing(pred: torch.Tensor,label: torch.Tensor,p: float=0.5):
        mask=torch.rand(pred.size(),device=pred.device)<p
        return torch.where(mask,label,pred)

class EarlyStopping:
    def __init__(self,patience=1):
        self.patience=patience
        self.patience_count=0
        self.prev_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,val_loss:float,model:torch.nn.Module):
        if self.prev_loss==np.inf:
            self.prev_loss=val_loss
            self.best_state={k: v.clone() for k,v in model.state_dict().items()}
            return None
        else:
            if not np.isfinite(val_loss):
                print(f"Loss is NaN or Inf!")
                self.early_stop=True
                model.load_state_dict(self.best_state)
                return model
            
            if self.prev_loss<=val_loss:
                self.patience_count+=1
                if self.patience<self.patience_count:
                    print(f"Loss increases during {self.patience_count} patience!")
                    self.early_stop=True
                    model.load_state_dict(self.best_state)
                    return model
            else:
                self.patience_count=0
                self.prev_loss=val_loss
                self.best_state={k: v.clone() for k,v in model.state_dict().items()}
                return None