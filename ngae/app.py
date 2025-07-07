import networkx as nx
import torch
from graph_utils import GraphUtils
from torch_geometric.utils import sort_edge_index


graph=nx.DiGraph()
graph.add_edge(0,1,w=1)
graph.add_edge(0,2,w=1)
graph.add_edge(1,3,w=2)
graph.add_edge(2,3,w=3)
graph.add_edge(0,3,w=4.5)
graph.add_edge(3,4,w=0.5)

graph.add_edge(0,0,w=6)
graph.add_edge(1,1,w=6)
graph.add_edge(2,2,w=6)
graph.add_edge(3,3,w=6)
graph.add_edge(4,4,w=6)

source_id=0

bfs_Q=GraphUtils.GraphAlgorithm.compute_BFS_step(graph=graph,source_id=source_id,init=True)
bfs_trajectory_tensor_list=[]
bfs_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bfs'))

bf_Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=source_id,init=True)
bf_trajectory_tensor_list=[]
p_trajectory_tensor_list=[]
p_idx_trajectory_tensor_list=[]
bf_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bf'))
p_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p'))
p_idx_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p_idx'))

"""
compute BFS, Bellman-Ford trajectory: [seq_len,num_nodes,1]
"""
while(True):
    if not bfs_Q and not bf_Q:
        break
    if bfs_Q:
        bfs_Q=GraphUtils.GraphAlgorithm.compute_BFS_step(graph=graph,source_id=source_id,Q=bfs_Q)
        bfs_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bfs'))
    if bf_Q:
        bf_Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=source_id,Q=bf_Q)
        bf_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bf'))
        p_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p'))
        GraphUtils.GraphManager.remap_node_predecessor_to_sorted_index(graph=graph)
        p_idx_trajectory_tensor_list.append(GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p_idx'))

bfs_trajectory=torch.stack(bfs_trajectory_tensor_list,dim=0)
bf_trajectory=torch.stack(bf_trajectory_tensor_list,dim=0)
p_trajectory=torch.stack(p_trajectory_tensor_list,dim=0)
p_idx_trajectory=torch.stack(p_idx_trajectory_tensor_list,dim=0)

"""
compute tau: [seq_len,1]
"""
bfs_tau=torch.cat([torch.ones(len(bfs_trajectory_tensor_list)-1,1),torch.zeros(1,1)],dim=0)
bf_tau=torch.cat([torch.ones(len(bf_trajectory_tensor_list)-1,1),torch.zeros(1,1)],dim=0)




print(bfs_trajectory.shape)
print(bfs_trajectory)