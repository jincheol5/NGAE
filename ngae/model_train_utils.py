import numpy as np
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
    def compute_predecessor_idx_from_edge_score_old(edge_score: torch.Tensor,edge_index: torch.Tensor,num_nodes: int):
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

    @staticmethod
    def compute_predecessor_idx_from_edge_score(edge_score: torch.Tensor,edge_index: torch.Tensor,num_nodes: int):
        scores=edge_score.view(-1) # [E]
        dst=edge_index[1] # [E]
        device=scores.device

        node_ids=torch.arange(num_nodes,device=device).unsqueeze(1) # [N,1]
        mask=(node_ids==dst.unsqueeze(0)) # [N,E]

        # global argmax: mask=False 자리엔 -inf 넣어 선택되지 않도록
        neg_inf=torch.finfo(scores.dtype).min
        masked_scores=torch.where(mask,scores.unsqueeze(0), neg_inf) # [N,E]
        global_idx=masked_scores.argmax(dim=1) # [N]

        # 각 True 위치에 대해 “로컬 순서” 만들기: cumsum-1
        # 예: mask[i] = [0,1,1,0,1] → cumsum = [0,1,2,2,3] → rank = cumsum-1 = [-1,0,1,1,2]
        # 로컬 엣지 0번째 → rank=0, 1번째→1, 2번째→2
        rank_matrix=mask.cumsum(dim=1)-1 # [N,E]

        # global_idx 위치에서 local rank 뽑기
        local_idx=rank_matrix.gather(1,global_idx.unsqueeze(1)) # [N,1]
        return local_idx

class EarlyStopping:
    """
    Args:
        patience
    """
    def __init__(self,patience=3):
        self.patience=patience
        self.prev_loss=np.inf
        self.prev_state = None
        self.early_stop=False
    def __call__(self,val_loss:float,model:torch.nn.Module):
        if self.prev_loss==np.inf:
            self.prev_loss=val_loss
            self.prev_state={k: v.clone() for k,v in model.state_dict().items()}
            return None
        
        if self.prev_loss<val_loss:
            model.load_state_dict(self.prev_state)
            return model
        
        if val_loss<self.prev_loss:
            self.prev_loss=val_loss
            self.prev_state={k: v.clone() for k,v in model.state_dict().items()}
            return None