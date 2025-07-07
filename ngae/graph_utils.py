import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch_geometric.utils import sort_edge_index
from torch_geometric.data import Data

class GraphUtils:
    class GraphManager:
        @staticmethod
        def set_self_loop(graph: nx.Graph):
            self_loop_edge_list=[(node,node) for node in graph.nodes()] 
            graph.add_edges_from(self_loop_edge_list)
        
        @staticmethod
        def set_edge_weight_attr(graph: nx.Graph):
            for edge in graph.edges():
                graph.edges[edge]['w']=np.float32(random.uniform(0.2,1.0))

        @staticmethod
        def initialize_node_attr_for_BFS(graph: nx.DiGraph,source_id: int=0):
            for node in graph.nodes():
                if node==source_id:
                    graph.nodes[node]['bfs']=1.0
                else:
                    graph.nodes[node]['bfs']=0.0

        @staticmethod
        def initialize_node_attr_for_BF(graph: nx.DiGraph,source_id: int=0):
            longest_shortest_path_len=float(graph.number_of_nodes())
            for node in graph.nodes():
                if node==source_id:
                    graph.nodes[node]['bf']=0.0
                else:
                    graph.nodes[node]['bf']=longest_shortest_path_len+1.0
                graph.nodes[node]['p']=node
            GraphUtils.GraphManager.remap_node_predecessor_to_sorted_index(graph=graph)

        @staticmethod
        def get_node_attr_tensor(graph: nx.DiGraph,attr: str='x'):
            match attr:
                case 'x':
                    group_attrs=['bfs','bf']
                    attr_dict={key: np.array([graph.nodes[node_id][key] for node_id in range(graph.number_of_nodes())],dtype=np.float32) for key in group_attrs}
                    attr_tensor=torch.tensor(np.column_stack([attr_dict[key] for key in group_attrs]),dtype=torch.float32)
                case 'bfs'|'bf':
                    attr_array=np.array([graph.nodes[node_id][attr] for node_id in range(graph.number_of_nodes())],dtype=np.float32)
                    attr_tensor=torch.tensor(attr_array,dtype=torch.float32)
                    attr_tensor=attr_tensor.unsqueeze(-1)
                case 'p'|'p_idx':
                    attr_array=np.array([graph.nodes[node_id][attr] for node_id in range(graph.number_of_nodes())],dtype=np.int64)
                    attr_tensor=torch.tensor(attr_array,dtype=torch.int64)
                    attr_tensor=attr_tensor.unsqueeze(-1)
            return attr_tensor

        @staticmethod
        def get_sorted_edge_index_tensor(graph: nx.DiGraph):
            edge_list=list(graph.edges()) 
            edge_index=torch.tensor(edge_list,dtype=torch.long).t().contiguous()
            sorted_edge_index=sort_edge_index(edge_index=edge_index,sort_by_row=False)
            return sorted_edge_index

        @staticmethod
        def get_edge_w_tensor(graph: nx.DiGraph,edge_index: torch.Tensor):
            edge_w=[graph[u.item()][v.item()]['w'] for u,v in zip(edge_index[0],edge_index[1])]
            edge_w=torch.tensor(edge_w)
            return edge_w.unsqueeze(-1)

        @staticmethod
        def remap_node_predecessor_to_sorted_index(graph: nx.DiGraph):
            for node in graph.nodes():
                pred=graph.nodes[node]['p']
                neighbor_sorted=sorted(graph.predecessors(node))
                graph.nodes[node]['p_idx']=neighbor_sorted.index(pred)

    class GraphGenerator:
        @staticmethod
        def generate_7_type_graphs(num_graphs: int,num_nodes: int):
            """
            <<Generate 7-type graphs>>
            1. ladder graph
            2. 2D grid graph
            3. tree graph
            4. Erdos-Renyi graph
            5. Barabasi-Albert graph
            6. 4-community graph
            7. 4-caveman graph
            """

            ladder_graph_list=[]
            grid_graph_list=[]
            tree_graph_list=[]
            Erdos_Renyi_graph_list=[]
            Barabasi_Albert_graph_list=[]
            community_graph_list=[]
            caveman_graph_list=[]

            # generate
            for _ in tqdm(range(num_graphs),desc=f"generate graph..."):
                """
                1. generate ladder graph
                """
                if num_nodes%2!=0:
                    raise ValueError("ladder graph requires an even number of nodes.")
                ladder_graph=nx.ladder_graph(num_nodes//2)
                GraphUtils.GraphManager.set_self_loop(graph=ladder_graph)
                GraphUtils.GraphManager.set_edge_weight_attr(graph=ladder_graph)
                ladder_graph_list.append(ladder_graph.to_directed())

                """
                2. generate 2D grid graph
                """
                side_length=int(np.ceil(np.sqrt(num_nodes)))
                grid_graph=nx.grid_2d_graph(side_length,side_length)
                grid_graph=nx.convert_node_labels_to_integers(grid_graph)
                grid_graph=grid_graph.subgraph(range(num_nodes)).copy()
                GraphUtils.GraphManager.set_self_loop(graph=grid_graph)
                GraphUtils.GraphManager.set_edge_weight_attr(graph=grid_graph)
                grid_graph_list.append(grid_graph.to_directed())
                
                """
                3. generate tree graph
                """
                tree_graph=nx.random_tree(num_nodes)
                GraphUtils.GraphManager.set_self_loop(graph=tree_graph)
                GraphUtils.GraphManager.set_edge_weight_attr(graph=tree_graph)
                tree_graph_list.append(tree_graph.to_directed())

                """
                4. generate Erdos-Renyi graph
                """
                p=min(np.log2(num_nodes)/num_nodes,0.5)
                erdos_renyi_graph=nx.erdos_renyi_graph(num_nodes,p)
                GraphUtils.GraphManager.set_self_loop(graph=erdos_renyi_graph)
                GraphUtils.GraphManager.set_edge_weight_attr(graph=erdos_renyi_graph)
                Erdos_Renyi_graph_list.append(erdos_renyi_graph.to_directed())

                """
                5. generate Barabasi-Albert graph
                """
                if num_nodes<=4:
                    raise ValueError("barabasi_albert graph requires more than 4 number of nodes.")
                m=random.choice([4,5])
                barabasi_albert_graph=nx.barabasi_albert_graph(num_nodes,m)
                GraphUtils.GraphManager.set_self_loop(graph=barabasi_albert_graph)
                GraphUtils.GraphManager.set_edge_weight_attr(graph=barabasi_albert_graph)
                Barabasi_Albert_graph_list.append(barabasi_albert_graph.to_directed())
                
                """
                6. generate 4 community graph
                """
                if num_nodes<4:
                    raise ValueError("4-Community graph requires at least 4 nodes.")
                community_size=num_nodes//4
                remaining_nodes=num_nodes%4
                communities=[nx.erdos_renyi_graph(community_size,0.7) for _ in range(4)]
                community_graph=nx.disjoint_union_all(communities)
                for i in range(remaining_nodes):
                    community_graph.add_node(community_graph.number_of_nodes())
                nodes=list(community_graph.nodes())
                for i in range(len(nodes)):
                    for j in range(i+1,len(nodes)):
                        if (i//community_size)!=(j//community_size):
                            if random.random()<0.01:
                                community_graph.add_edge(i,j)
                GraphUtils.GraphManager.set_self_loop(graph=community_graph)
                GraphUtils.GraphManager.set_edge_weight_attr(graph=community_graph)
                community_graph_list.append(community_graph.to_directed())
            
                """
                7. generate 4-caveman graph
                """
                if num_nodes<4:
                    raise ValueError("4-Caveman graph requires at least 4 nodes.")
                clique_size=num_nodes//4
                remaining_nodes=num_nodes%4
                caveman_graph=nx.caveman_graph(4,clique_size)
                for i in range(remaining_nodes):
                    caveman_graph.add_node(caveman_graph.number_of_nodes())
                edges_to_remove=[edge for edge in caveman_graph.edges() if random.random()<0.7]
                caveman_graph.remove_edges_from(edges_to_remove)
                num_shortcuts=int(0.025*num_nodes)
                for _ in range(num_shortcuts):
                    u,v=random.sample(list(caveman_graph.nodes()),2)
                    if not caveman_graph.has_edge(u,v):
                        caveman_graph.add_edge(u,v)
                GraphUtils.GraphManager.set_self_loop(graph=caveman_graph)
                GraphUtils.GraphManager.set_edge_weight_attr(graph=caveman_graph)
                caveman_graph_list.append(caveman_graph.to_directed())

            graph_list_dict={}
            graph_list_dict['ladder']=ladder_graph_list
            graph_list_dict['grid']=grid_graph_list
            graph_list_dict['tree']=tree_graph_list
            graph_list_dict['erdos_renyi']=Erdos_Renyi_graph_list
            graph_list_dict['barabasi_albert']=Barabasi_Albert_graph_list
            graph_list_dict['community']=community_graph_list
            graph_list_dict['caveman']=caveman_graph_list

            return graph_list_dict

    class GraphVisualizer:
        @staticmethod
        def visualize_graph(graph: nx.DiGraph):
            # layout 계산
            pos=nx.kamada_kawai_layout(graph)

            # node, edge 그리기
            plt.figure(figsize=(10,10))
            nx.draw_networkx_nodes(graph,pos,node_size=100,node_color='lightgreen',edgecolors='black')
            nx.draw_networkx_edges(graph,pos,width=2,edge_color='gray',arrows=True,arrowsize=20)

            # 축 제거 및 출력
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    class GraphAlgorithm:
        @staticmethod
        def compute_BFS_step(graph: nx.DiGraph,source_id: int=0,init: bool=False,Q: set=None):
            Q_next=set()
            if init:
                GraphUtils.GraphManager.initialize_node_attr_for_BFS(graph=graph,source_id=source_id)
                Q_next.add(source_id)
            else:
                for src in Q:
                    for _,tar in graph.edges(src):
                        if graph.nodes[tar]['bfs']==0.0:
                            graph.nodes[tar]['bfs']=1.0
                            Q_next.add(tar)
            return Q_next

        @staticmethod
        def compute_BF_step(graph: nx.DiGraph,source_id: int=0,init: bool=False,Q: set=None):
            Q_next=set()
            if init:
                GraphUtils.GraphManager.initialize_node_attr_for_BF(graph=graph,source_id=source_id)
                Q_next.add(source_id)
            else:
                for src in Q:
                    for _,tar in graph.edges(src):
                        if graph.nodes[src]['bf']+graph.edges[(src,tar)]['w']<graph.nodes[tar]['bf']:
                            graph.nodes[tar]['bf']=graph.nodes[src]['bf']+graph.edges[(src,tar)]['w']
                            graph.nodes[tar]['p']=src
                            Q_next.add(tar)
            return Q_next
    
    class GraphProcessor:
        @staticmethod
        def algo_trajectory_to_PyG_Data(graph: nx.DiGraph,graph_type: str="basic",graph_id: int=0,source_id: int=0):
            """
            initialize
            """
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

            """
            convert to PyG Data
            """
            edge_index=GraphUtils.GraphManager.get_sorted_edge_index_tensor(graph=graph)
            edge_attr=GraphUtils.GraphManager.get_edge_w_tensor(graph=graph,edge_index=edge_index)
            data=Data(edge_index=edge_index,edge_attr=edge_attr,num_nodes=graph.number_of_nodes())

            # meta data
            data.source_id=source_id
            data.graph_type=graph_type
            data.graph_num=graph_id

            # BFS
            data.bfs=bfs_trajectory
            data.bfs_tau=bfs_tau

            # Bellman-ford
            data.bf=bf_trajectory
            data.p=p_trajectory
            data.p_idx=p_idx_trajectory
            data.bf_tau=bf_tau

            return data

        @staticmethod
        def graph_list_dict_to_PyG_Data_list(graph_list_dict: dict,file_name: str="train_20_nodes"):
            all_data_list=[]
            ladder_data_list=[]
            grid_data_list=[]
            tree_data_list=[]
            erdos_renyi_data_list=[]
            barabasi_albert_data_list=[]
            community_data_list=[]
            caveman_data_list=[]
            for graph_type,graph_list in tqdm(graph_list_dict.items(),desc=f"Convert {file_name} graph_list_dict to PyG_Data_list..."):
                for graph_id,graph in enumerate(graph_list):
                    data_list=[]
                    for source_id in graph.nodes():
                        data_list.append(GraphUtils.GraphProcessor.algo_trajectory_to_PyG_Data(graph=graph,graph_type=graph_type,graph_id=graph_id,source_id=source_id))
                    all_data_list+=data_list
                    match graph_type:
                        case 'ladder':
                            ladder_data_list+=data_list
                        case 'grid':
                            grid_data_list+=data_list
                        case 'tree':
                            tree_data_list+=data_list
                        case 'erdos_renyi':
                            erdos_renyi_data_list+=data_list
                        case 'barabasi_albert':
                            barabasi_albert_data_list+=data_list
                        case 'community':
                            community_data_list+=data_list
                        case 'caveman':
                            caveman_data_list+=data_list
            
            data_list_dict={}
            data_list_dict['all']=all_data_list
            data_list_dict['ladder']=ladder_data_list
            data_list_dict['grid']=grid_data_list
            data_list_dict['tree']=tree_data_list
            data_list_dict['erdos_renyi']=erdos_renyi_data_list
            data_list_dict['barabasi_albert']=barabasi_albert_data_list
            data_list_dict['community']=community_data_list
            data_list_dict['caveman']=caveman_data_list

            return data_list_dict