import torch
import torch.nn.functional as F

class Metrics:
    @staticmethod
    def compute_BF_loss(pred: torch.Tensor, label: torch.Tensor):
        """
        Input:
            -pred: [seq_len-1,N,1]
            -label: [seq_len-1,N,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        total_loss=torch.zeros((),device=pred.device) 
        seq_len=pred.size(0)
        for i in range(seq_len):
            total_loss=total_loss+F.mse_loss(input=pred[i],target=label[i])
        return total_loss

    @staticmethod
    def compute_predecessor_loss(pred: torch.Tensor, label: torch.Tensor):
        """
        Input:
            -pred: [seq_len-1,E,1]
            -label: [seq_len-1,N,1]
        Output:
            -loss scalar tensor: [] (0차원)
        """
        seq_len,num_nodes,_=label.size()