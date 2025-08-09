import numpy as np
import torch
import torch.nn.functional as F

class ModelTrainUtils:
    @staticmethod
    def convert_edge_score_to_softmax_one_hot_p(edge_score: torch.Tensor,edge_index: torch.Tensor,num_nodes:int):
        logits=torch.full((num_nodes,num_nodes),fill_value=-float('inf'),device=edge_score.device,dtype=torch.float32)
        edge_score=edge_score.squeeze(-1)
        src,tar=edge_index  
        logits[tar,src]=edge_score
        probs=F.softmax(logits,dim=1) # target 노드(행) 기준 softmax 적용
        return probs.unsqueeze(-1) # [N,N,1]

    @staticmethod
    def teacher_forcing(pred: torch.Tensor,label: torch.Tensor,p: float=0.5):
        mask=torch.rand_like(pred)<p    
        return torch.where(mask,label,pred)

    @staticmethod
    def compute_r_from_logit(logit: torch.Tensor):
        prob=F.sigmoid(logit)           
        mask=prob>=0.5                   
        return mask.to(logit.dtype)


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