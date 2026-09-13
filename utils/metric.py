import torch

class Metric:
    @staticmethod
    def compute_reachability_accuracy(
            r_traj:torch.Tensor,
            pred_r_traj:torch.Tensor,
            bfs_mask:torch.Tensor
        ):
        """
        현재 state와 다음 state가 둘 다 실제 데이터에 존재하는 transition만 acc에 포함
        
        Input:
            r_traj: [B,max_bfs_len,N]
            pred_r_traj: [B,max_bfs_len-1,N]
            bfs_mask: [B,max_bfs_len]
        """
        # get label
        label_r_traj=r_traj[:,1:] # [B,max_bfs_len-1,N]

        # 현재 step과 다음 step이 모두 존재하는 transition만 사용
        valid_mask=(bfs_mask[:,:-1] & bfs_mask[:,1:]) # [B,max_bfs_len-1]

        # graph-level step mask -> node-level mask
        valid_mask=valid_mask.unsqueeze(-1).expand_as(label_r_traj) # [B,max_bfs_len-1,N]

        # valid prediction / label만 추출
        valid_logits=pred_r_traj[valid_mask]
        valid_labels=label_r_traj[valid_mask].bool()

        # logit -> probability
        valid_probs=torch.sigmoid(valid_logits)

        # probability -> binary prediction
        pred_labels=valid_probs>=0.5

        # accuracy
        accuracy=(
            pred_labels==valid_labels
        ).float().mean().item()
        return accuracy