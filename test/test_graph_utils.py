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
            Test 3. remap_node_predecessor_attr_to_sorted_index()
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
                GraphUtils.GraphManager.remap_node_predecessor_attr_to_sorted_index(graph=graph)
                p_idx_step=np.array([graph.nodes[node]['p_idx'] for node in sorted(graph.nodes())]).reshape(1,-1)
                p_idx=np.append(p_idx,p_idx_step,axis=0)
            print(f"Bellman-Ford Predecessor:")
            for step in p.T:
                print(*step)
            print()
            print(f"Bellman-Ford Predecessor index:")
            for step in p_idx.T:
                print(*step)


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
