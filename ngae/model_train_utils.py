import torch
import torch.nn.functional as F

class ModelTrainUtils:
    @staticmethod
    def teacher_forcing(pred: torch.Tensor,label: torch.Tensor,p: float=0.5):
        mask=torch.rand_like(pred)<p    
        return torch.where(mask,label,pred)

    @staticmethod
    def compute_BFS_from_logit(logit: torch.Tensor,threshold: float=0.5):
        prob=F.sigmoid(logit)           
        mask=prob>=threshold                   
        return mask.to(logit.dtype)

    @staticmethod
    def compute_predecessor_idx_from_edge_score(edge_score: torch.Tensor,edge_index: torch.Tensor,num_nodes: int):
        p_idx=torch.full((num_nodes,),-1,dtype=torch.long,device=edge_score.device)
        edge_score=edge_score.view(-1) # [E,]
        dst=edge_index[1]

        for target_node in range(num_nodes):
            mask=(dst==target_node)
            if not mask.any():
                continue
            incoming_score=edge_score[mask]
            prob=F.softmax(incoming_score,dim=0)
            p_idx[target_node]=prob.argmax()

        return p_idx.unsqueeze(1)