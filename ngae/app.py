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

GraphUtils.GraphManager.initialize_node_attr_for_BFS(graph=graph,source_id=0)
GraphUtils.GraphManager.initialize_node_attr_for_BF(graph=graph,source_id=0)

bfs_tensor=GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bfs')
print(f"BFS tensor:")
print(bfs_tensor)
print()

bf_tensor=GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='bf')
print(f"BF tensor:")
print(bf_tensor)
print()

x_tensor=GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='x')
print(f"x tensor:")
print(x_tensor)
print()

p_tensor=GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p')
print(f"p tensor:")
print(p_tensor)
print()

p_idx_tensor=GraphUtils.GraphManager.get_node_attr_tensor(graph=graph,attr='p_idx')
print(f"p_idx tensor:")
print(p_idx_tensor)
print()