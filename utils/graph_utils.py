import random
import numpy as np
import networkx as nx
import torch
from typing import Literal
from tqdm import tqdm
from torch_geometric.utils import sort_edge_index

class GraphUtils:
    @staticmethod
    def remove_self_loop(graph:nx.Graph):
        graph.remove_edges_from(nx.selfloop_edges(graph))

    @staticmethod
    def set_edge_weight(graph:nx.Graph):
        for edge in graph.edges():
            graph.edges[edge]["w"]=np.float32(random.uniform(0.2,1.0))

    @staticmethod
    def init_node_state(
            graph:nx.Graph,
            source:int
        ):
        max_d=float(graph.number_of_nodes())
        for node in graph.nodes():
            if node==source:
                graph.nodes[node]["r"]=True
                graph.nodes[node]["d"]=0.0
            else:
                graph.nodes[node]["r"]=False
                graph.nodes[node]["d"]=max_d
            graph.nodes[node]["p"]=node

    @staticmethod
    def get_node_state_tensor(
            graph:nx.Graph,
            state:Literal["r","d","p"]
        ):
        match state:
            case "r":
                tensor_dtype=torch.bool
            case "d":
                tensor_dtype=torch.float32
            case "p":
                tensor_dtype=torch.int16
        state_tensor=torch.tensor(
            [
                graph.nodes[node_id][state] 
                for node_id in range(graph.number_of_nodes())
            ],
            dtype=tensor_dtype
        )
        return state_tensor

    @staticmethod
    def get_edge_index_tensor(graph:nx.Graph):
        edge_list=list(graph.edges()) 
        # 역방향 edge도 추가
        edge_list+=[
            (v,u) for u,v in graph.edges()
            if u!=v
        ]
        edge_index=torch.tensor(edge_list,dtype=torch.long).t().contiguous()
        sorted_edge_index=sort_edge_index(edge_index=edge_index,sort_by_row=False)
        return sorted_edge_index

    @staticmethod
    def get_edge_weight_tensor(
            graph:nx.Graph,
            edge_index:torch.Tensor
        ):
        edge_weight=[
            graph[u.item()][v.item()]["w"] 
            for u,v in zip(edge_index[0],edge_index[1])
        ]
        edge_weight=torch.tensor(edge_weight)
        return edge_weight.unsqueeze(-1)

class GraphGenerator:
    @staticmethod
    def generate_7_type_graphs(
            n_graph:int, 
            n_node:int
        ):
        """
        << Generate 7-type graphs >>
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
        erdos_renyi_graph_list=[]
        barabasi_albert_graph_list=[]
        community_graph_list=[]
        caveman_graph_list=[]

        ### generate graph
        for _ in tqdm(range(n_graph),desc=f"Generate Graph..."):
            # 1. generate ladder graph
            if n_node%2!=0:
                raise ValueError(f"ladder graph requires an even number of nodes.")
            ladder_graph=nx.ladder_graph(n_node//2)
            GraphUtils.remove_self_loop(graph=ladder_graph)
            GraphUtils.set_edge_weight(graph=ladder_graph)
            ladder_graph_list.append(ladder_graph)

            # 2. generate 2D grid graph
            side_length=int(np.ceil(np.sqrt(n_node)))
            grid_graph=nx.grid_2d_graph(side_length,side_length)
            grid_graph=nx.convert_node_labels_to_integers(grid_graph)
            grid_graph=grid_graph.subgraph(range(n_node)).copy()
            GraphUtils.remove_self_loop(graph=grid_graph)
            GraphUtils.set_edge_weight(graph=grid_graph)
            grid_graph_list.append(grid_graph)

            # 3. generate tree graph
            tree_graph=nx.random_labeled_tree(n_node)
            GraphUtils.remove_self_loop(graph=tree_graph)
            GraphUtils.set_edge_weight(graph=tree_graph)
            tree_graph_list.append(tree_graph)

            # 4. generate Erdos-Renyi graph
            p=min(np.log2(n_node)/n_node,0.5)
            erdos_renyi_graph=nx.erdos_renyi_graph(n_node,p)
            GraphUtils.remove_self_loop(graph=erdos_renyi_graph)
            GraphUtils.set_edge_weight(graph=erdos_renyi_graph)
            erdos_renyi_graph_list.append(erdos_renyi_graph)

            # 5. generate Barabasi-Albert graph
            if n_node<=4:
                raise ValueError("Barabasi-Albert graph requires more than 4 number of nodes.")
            m=random.choice([4,5])
            barabasi_albert_graph=nx.barabasi_albert_graph(n_node,m)
            GraphUtils.remove_self_loop(graph=barabasi_albert_graph)
            GraphUtils.set_edge_weight(graph=barabasi_albert_graph)
            barabasi_albert_graph_list.append(barabasi_albert_graph)

            # 6. generate 4 community graph
            if n_node<4:
                raise ValueError("4-Community graph requires at least 4 nodes.")
            community_size=n_node//4
            remaining_nodes=n_node%4
            communities=[nx.erdos_renyi_graph(community_size,0.1) for _ in range(4)]
            community_graph=nx.disjoint_union_all(communities)
            for i in range(remaining_nodes):
                community_graph.add_node(community_graph.number_of_nodes())
            nodes=list(community_graph.nodes())
            for i in range(len(nodes)):
                for j in range(i+1,len(nodes)):
                    if (i//community_size)!=(j//community_size):
                        if random.random()<0.01:
                            community_graph.add_edge(i,j)
            GraphUtils.remove_self_loop(graph=community_graph)
            GraphUtils.set_edge_weight(graph=community_graph)
            community_graph_list.append(community_graph)

            # 7. generate 4-caveman graph
            if n_node<4:
                raise ValueError("4-Caveman graph requires at least 4 nodes.")
            clique_size=n_node//4
            remaining_nodes=n_node%4
            caveman_graph=nx.caveman_graph(4,clique_size)
            for i in range(remaining_nodes):
                caveman_graph.add_node(caveman_graph.number_of_nodes())
            edges_to_remove=[edge for edge in caveman_graph.edges() if random.random()<0.8]
            caveman_graph.remove_edges_from(edges_to_remove)
            num_shortcuts=int(0.025*n_node)
            for _ in range(num_shortcuts):
                u,v=random.sample(list(caveman_graph.nodes()),2)
                if not caveman_graph.has_edge(u,v):
                    caveman_graph.add_edge(u,v)
            GraphUtils.remove_self_loop(graph=caveman_graph)
            GraphUtils.set_edge_weight(graph=caveman_graph)
            caveman_graph_list.append(caveman_graph)
        return {
            "ladder":ladder_graph_list,
            "grid":grid_graph_list,
            "tree":tree_graph_list,
            "erdos_renyi":erdos_renyi_graph_list,
            "barabasi_albert":barabasi_albert_graph_list,
            "community":community_graph_list,
            "caveman":caveman_graph_list
        }


class GraphAlgorithm:
    @staticmethod
    def compute_BFS_step(
            graph:nx.Graph,
            Q:set
        ):
        """
        """
        Q_next=set()
        for src in Q:
            for tar in graph.neighbors(src):
                if graph.nodes[tar]["r"]==False:
                    graph.nodes[tar]["r"]=True
                    Q_next.add(tar)
        return Q_next

    @staticmethod
    def compute_BF_step(
            graph:nx.Graph,
            Q:set
        ):
        """
        """
        Q_next=set()
        for src in Q:
            for tar in graph.neighbors(src):
                if graph.nodes[src]["d"]+graph.edges[(src,tar)]["w"]<graph.nodes[tar]["d"]:
                    graph.nodes[tar]["d"]=graph.nodes[src]["d"]+graph.edges[(src,tar)]["w"]
                    graph.nodes[tar]["p"]=src
                    Q_next.add(tar)
        return Q_next