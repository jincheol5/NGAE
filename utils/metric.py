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

    @staticmethod
    def compute_predecessor_accuracy(
            p_traj:torch.Tensor,
            edge_score_traj:torch.Tensor,
            edge_index:torch.Tensor,
            bf_mask:torch.Tensor
        ):
        """
        유효 transition의 모든 node에 대한 predecessor 정확도.
        loss와 동일하게 incoming neighbor와 점수 0인 자기 자신이 후보.
        유효 transition이 없으면 0.0을 반환한다.

        Input:
            p_traj: [B,T,N], local predecessor node id
            edge_score_traj: [E,T-1,1], 다음 predecessor의 raw logits
            edge_index: [2,E], source -> target, self-loop와 중복 엣지 없음
            bf_mask: [B,T], 실제 step은 True
        """
        batch_size,n_step,n_node=p_traj.shape
        valid_mask=bf_mask[:,:-1] & bf_mask[:,1:]
        if not valid_mask.any():
            return 0.0

        # 다음 step의 local predecessor 정답
        labels=p_traj[:,1:].long()

        # shape: [graph, step, target, predecessor 후보]
        logits=edge_score_traj.new_full(
            (batch_size,n_step-1,n_node,n_node),float('-inf')
        )
        nodes=torch.arange(n_node,device=edge_score_traj.device)
        logits[:,:,nodes,nodes]=0.0
        src,tar=edge_index
        logits[tar//n_node,:,tar%n_node,src%n_node]=edge_score_traj[:,:,0]

        # softmax 없이 argmax로 가장 높은 점수의 후보 선택
        pred_labels=logits[valid_mask].argmax(dim=-1)
        accuracy=(pred_labels==labels[valid_mask]).float().mean().item()
        return accuracy
