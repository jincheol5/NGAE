import argparse
import networkx as nx
import numpy as np
from ngae import GraphUtils

def test(config: dict):
    match config["test_num"]:
        case 0:
            pass
        case 1:
            """
            Test 1. compute_BFS_step()
            """
            graph=nx.DiGraph()
            graph.add_edge(0,1,w=1)
            graph.add_edge(0,2,w=1)
            graph.add_edge(1,3,w=2)
            graph.add_edge(2,3,w=3)
            graph.add_edge(0,3,w=4.5)
            graph.add_edge(3,4,w=0.5)

            bfs=np.empty((0,5))
            Q=GraphUtils.GraphAlgorithm.compute_BFS_step(graph=graph,source_id=0,init=True)
            bfs_step=np.array([graph.nodes[node]['bfs'] for node in sorted(graph.nodes())]).reshape(1,-1)
            bfs=np.append(bfs,bfs_step,axis=0)
            while Q:
                Q=GraphUtils.GraphAlgorithm.compute_BFS_step(graph=graph,source_id=0,Q=Q)
                bfs_step=np.array([graph.nodes[node]['bfs'] for node in sorted(graph.nodes())]).reshape(1,-1)
                bfs=np.append(bfs,bfs_step,axis=0)
            print(f"BFS Reachability:")
            for bfs_step in bfs.T:
                print(*bfs_step)

        case 2:
            """
            Test 2. compute_BF_step()
            """
            graph=nx.DiGraph()
            graph.add_edge(0,1,w=1)
            graph.add_edge(0,2,w=1)
            graph.add_edge(1,3,w=2)
            graph.add_edge(2,3,w=3)
            graph.add_edge(0,3,w=4.5)
            graph.add_edge(3,4,w=0.5)

            bf=np.empty((0,5))
            p=np.empty((0,5))
            Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=0,init=True)
            bf_step=np.array([graph.nodes[node]['bf'] for node in sorted(graph.nodes())]).reshape(1,-1)
            bf=np.append(bf,bf_step,axis=0)
            p_step=np.array([graph.nodes[node]['p'] for node in sorted(graph.nodes())]).reshape(1,-1)
            p=np.append(p,p_step,axis=0)
            while Q:
                Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=0,Q=Q)
                bf_step=np.array([graph.nodes[node]['bf'] for node in sorted(graph.nodes())]).reshape(1,-1)
                bf=np.append(bf,bf_step,axis=0)
                p_step=np.array([graph.nodes[node]['p'] for node in sorted(graph.nodes())]).reshape(1,-1)
                p=np.append(p,p_step,axis=0)
            print(f"Bellman-Ford Distance:")
            for step in bf.T:
                print(*step)
            print()
            print(f"Bellman-Ford Predecessor:")
            for step in p.T:
                print(*step)

        case 3:
            """
            Test 3. get_node_attr_tensor()
            """
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


        case 4:
            """
            Test 4. get_sorted_edge_index_tensor()
            """
            graph=nx.DiGraph()
            graph.add_edge(0,1,w=1)
            graph.add_edge(0,2,w=1)
            graph.add_edge(1,3,w=2)
            graph.add_edge(2,3,w=3)
            graph.add_edge(0,3,w=4.5)
            graph.add_edge(3,4,w=0.5)

            edge_index=GraphUtils.GraphManager.get_sorted_edge_index_tensor(graph=graph)
            print(edge_index)
        
        case 5:
            """
            Test 5. get_edge_w_tensor()
            """
            graph=nx.DiGraph()
            graph.add_edge(0,1,w=1)
            graph.add_edge(0,2,w=1)
            graph.add_edge(1,3,w=2)
            graph.add_edge(2,3,w=3)
            graph.add_edge(0,3,w=4.5)
            graph.add_edge(3,4,w=0.5)

            edge_index=GraphUtils.GraphManager.get_sorted_edge_index_tensor(graph=graph)
            print(edge_index)

            edge_attr=GraphUtils.GraphManager.get_edge_w_tensor(graph=graph,edge_index=edge_index)
            print(edge_attr)

        case 6:
            """
            Test 6. remap_node_predecessor_attr_to_sorted_index()
            """
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

            p=np.empty((0,5))
            p_idx=np.empty((0,5))
            Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=0,init=True)
            p_step=np.array([graph.nodes[node]['p'] for node in sorted(graph.nodes())]).reshape(1,-1)
            p=np.append(p,p_step,axis=0)
            p_idx_step=np.array([graph.nodes[node]['p_idx'] for node in sorted(graph.nodes())]).reshape(1,-1)
            p_idx=np.append(p_idx,p_idx_step,axis=0)
            while Q:
                Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=0,Q=Q)
                p_step=np.array([graph.nodes[node]['p'] for node in sorted(graph.nodes())]).reshape(1,-1)
                p=np.append(p,p_step,axis=0)
                GraphUtils.GraphManager.remap_node_predecessor_to_sorted_index(graph=graph)
                p_idx_step=np.array([graph.nodes[node]['p_idx'] for node in sorted(graph.nodes())]).reshape(1,-1)
                p_idx=np.append(p_idx,p_idx_step,axis=0)
            print(f"Bellman-Ford Predecessor:")
            for step in p.T:
                print(*step)
            print()
            print(f"Bellman-Ford Predecessor index:")
            for step in p_idx.T:
                print(*step)

        case 7:
            """
            Test 7. algo_trajectory_to_PyG_Data()
            """
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

            data=GraphUtils.GraphProcessor.algo_trajectory_to_PyG_Data(graph=graph)

            print(f"edge_index:")
            print(data.edge_index)
            print()

            print(f"edge_attr:")
            print(data.edge_attr)
            print()

            print(f"BFS trajectory:")
            print(data.bfs)
            print()

            print(f"Bellman-Ford trajectory:")
            print(data.bf)
            print()

            print(f"Predecessor trajectory:")
            print(data.p)
            print()

            print(f"Predecessor idx trajectory:")
            print(data.p_idx)
            print()

            print(f"BFS tau:")
            print(data.bfs_tau)
            print()

            print(f"Bellman-Ford tau:")
            print(data.bf_tau)
            print()

"""
Execute Test
"""
parser=argparse.ArgumentParser()
parser.add_argument("--test_num",type=int,default=0)
args=parser.parse_args()

config={
    "test_num":args.test_num
}

test(config=config)
