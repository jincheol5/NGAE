# @staticmethod
#     def validate(model,val_graph_type,val_data_loader,config):
#         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model.to(device)
#         model.eval()

#         match config['task']:
#             case 'bfs':
#                 y_step_acc_list=[]
#                 y_last_acc_list=[]
#                 tau_step_acc_list=[]
#                 tau_last_acc_list=[]
#             case 'bf':
#                 p_step_acc_list=[]
#                 p_last_acc_list=[]
#                 tau_step_acc_list=[]
#                 tau_last_acc_list=[]

#         with torch.no_grad():
#             for batch in tqdm(val_data_loader,desc=f"Validate {val_graph_type} graph..."):
#                 batch=batch.to(device)
#                 h_0=torch.zeros((batch.num_nodes,config['latent_dim']),dtype=torch.float32)
#                 h_0=h_0.to(device)

#                 match config['task']:
#                     case 'bfs':
#                         algo_trajectory=batch.bfs # [seq_len,N,1]
#                         tau_seq_label=batch.bfs_tau # [seq_len-1,1]
#                     case 'bf':
#                         algo_trajectory=batch.bf # [seq_len,N,1]
#                         p_idx_trajectory=batch.p_idx # [seq_len,N,1]
#                         tau_seq_label=batch.bf_tau # [seq_len-1,1]

#                 output=model(algo_trajectory=algo_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,mode='test')
#                 y_seq=output['y'] # [seq_len-1,N,1]
#                 tau_seq=output['tau'] # [seq_len-1,1]
#                 if config['task']=='bf':
#                     edge_score_seq=output['edge_score'] # [seq_len-1,E,1]

#                 """
#                 compute acc
#                 """
#                 match config['task']:
#                     case 'bfs':
#                         batch_y_step_acc,batch_y_last_acc=Metrics.compute_BFS_seq_acc(logit=y_seq,label=algo_trajectory[1:])
#                         batch_tau_step_acc,batch_tau_last_acc=Metrics.compute_tau_seq_acc(logit=tau_seq,label=tau_seq_label)
#                         y_step_acc_list.append(batch_y_step_acc)
#                         y_last_acc_list.append(batch_y_last_acc)
#                         tau_step_acc_list.append(batch_tau_step_acc)
#                         tau_last_acc_list.append(batch_tau_last_acc)
#                     case 'bf':
#                         batch_p_step_acc,batch_p_last_acc=Metrics.compute_predecessor_seq_acc(logit=edge_score_seq,label=p_idx_trajectory[1:],edge_index=batch.edge_index)
#                         batch_tau_step_acc,batch_tau_last_acc=Metrics.compute_tau_seq_acc(logit=tau_seq,label=tau_seq_label)
#                         p_step_acc_list.append(batch_p_step_acc)
#                         p_last_acc_list.append(batch_p_last_acc)
#                         tau_step_acc_list.append(batch_tau_step_acc)
#                         tau_last_acc_list.append(batch_tau_last_acc)
        
#         match config['task']:
#             case 'bfs':
#                 y_step_acc=np.mean(y_step_acc_list)
#                 y_last_acc=np.mean(y_last_acc_list)
#                 tau_step_acc=np.mean(tau_step_acc_list)
#                 tau_last_acc=np.mean(tau_last_acc_list)

#                 print(f"Validate {val_graph_type} graph BFS step acc: {y_step_acc} last acc: {y_last_acc}")
#                 print(f"Validate {val_graph_type} graph tau step acc: {tau_step_acc} last acc: {tau_last_acc}")
#             case 'bf':
#                 p_step_acc=np.mean(p_step_acc_list)
#                 p_last_acc=np.mean(p_last_acc_list)
#                 tau_step_acc=np.mean(tau_step_acc_list)
#                 tau_last_acc=np.mean(tau_last_acc_list)

#                 print(f"Validate {val_graph_type} graph predecessor step acc: {p_step_acc} last acc: {p_last_acc}")
#                 print(f"Validate {val_graph_type} graph tau step acc: {tau_step_acc} last acc: {tau_last_acc}")