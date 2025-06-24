import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class GraphUtils:
    class GraphManager:
        @staticmethod
        def set_self_loop(graph: nx.Graph):
            self_loop_edge_list=[(node,node) for node in graph.nodes()] 
            graph.add_edges_from(self_loop_edge_list)
        
        @staticmethod
        def set_edge_weight_attr(graph: nx.Graph):
            for edge in graph.edges():
                graph.edges[edge]['weight']=np.float32(random.uniform(0.2,1.0))

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
                ladder_graph_list.append(ladder_graph)

                """
                2. generate 2D grid graph
                """
                side_length=int(np.ceil(np.sqrt(num_nodes)))
                grid_graph=nx.grid_2d_graph(side_length,side_length)
                grid_graph=nx.convert_node_labels_to_integers(grid_graph)
                grid_graph=grid_graph.subgraph(range(num_nodes)).copy()
                grid_graph_list.append(grid_graph)
                
                """
                3. generate tree graph
                """
                tree_graph=nx.random_tree(num_nodes)
                tree_graph_list.append(tree_graph)

                """
                4. generate Erdos-Renyi graph
                """
                p=min(np.log2(num_nodes)/num_nodes,0.5)
                erdos_renyi_graph=nx.erdos_renyi_graph(num_nodes,p)
                Erdos_Renyi_graph_list.append(erdos_renyi_graph)

                """
                5. generate Barabasi-Albert graph
                """
                if num_nodes<=4:
                    raise ValueError("barabasi_albert graph requires more than 4 number of nodes.")
                m=random.choice([4,5])
                barabasi_albert_graph=nx.barabasi_albert_graph(num_nodes,m)
                Barabasi_Albert_graph_list.append(barabasi_albert_graph)
                
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
                community_graph_list.append(community_graph)
            
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
                caveman_graph_list.append(caveman_graph)
        
            """
            set self-loop, edge w attr
            """
            for graph_list in [ladder_graph_list,grid_graph_list,tree_graph_list,Erdos_Renyi_graph_list,Barabasi_Albert_graph_list,community_graph_list,caveman_graph_list]:
                for graph in graph_list:
                    GraphUtils.GraphManager.set_self_loop(graph=graph)
                    GraphUtils.GraphManager.set_edge_weight_attr(graph=graph)
            
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
        def visualize_graph(graph: nx.Graph):
            # layout 계산
            pos=nx.kamada_kawai_layout(graph)

            # node, edge 그리기
            plt.figure(figsize=(10,10))
            nx.draw_networkx_nodes(graph,pos,node_size=100,node_color='lightgreen',edgecolors='black')
            nx.draw_networkx_edges(graph,pos,width=2,edge_color='gray')

            # 축 제거 및 출력
            plt.axis('off')
            plt.tight_layout()
            plt.show()
