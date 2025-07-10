import torch
import torch.nn.functional as F

class ModelTrainUtils:
    @staticmethod
    def teacher_forcing(pred: torch.Tensor,label: torch.Tensor,p: float=0.5):
        mask=torch.rand_like(pred)<p    
        return torch.where(mask,label,pred)

    @staticmethod
    def compute_BFS_from_logit(logit: torch.Tensor,threshold: float=0.5):
        prob=torch.sigmoid(logit)           
        mask=prob>=threshold                   
        return mask.to(logit.dtype)