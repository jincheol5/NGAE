import argparse
import networkx as nx
from ngae import GraphUtils

def test(test_number: int):
    match test_number:
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
            graph.add_edge(0,3,w=3.5)
            graph.add_edge(3,4,w=1.5)

            Q=GraphUtils.GraphAlgorithm.compute_BFS_step(graph=graph,source_id=0,init=True)
            print(f"Q: {Q}")
            for node in graph.nodes():
                print(f"{node} bfs: {graph.nodes[node]['bfs']}")
            print()
            while Q:
                Q=GraphUtils.GraphAlgorithm.compute_BFS_step(graph=graph,source_id=0,Q=Q)
                print(f"Q: {Q}")
                for node in graph.nodes():
                    print(f"{node} bfs: {graph.nodes[node]['bfs']}")
                print()

        case 2:
            """
            Test 2. compute_BF_step()
            """
            graph=nx.DiGraph()
            graph.add_edge(0,1,w=1)
            graph.add_edge(0,2,w=1)
            graph.add_edge(1,3,w=2)
            graph.add_edge(2,3,w=3)
            graph.add_edge(0,3,w=3.5)
            graph.add_edge(3,4,w=1.5)

            Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=0,init=True)
            print(f"Q: {Q}")
            for node in graph.nodes():
                print(f"{node} p: {graph.nodes[node]['p']} bf: {graph.nodes[node]['bf']}")
            print()
            while Q:
                Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=0,Q=Q)
                print(f"Q: {Q}")
                for node in graph.nodes():
                    print(f"{node} p: {graph.nodes[node]['p']} bf: {graph.nodes[node]['bf']}")
                print()

        case 3:
            """
            Test 3. remap_node_predecessor_attr_to_sorted_index()
            """
            graph=nx.DiGraph()
            graph.add_edge(0,1,w=1)
            graph.add_edge(0,2,w=1)
            graph.add_edge(1,3,w=2)
            graph.add_edge(2,3,w=3)
            graph.add_edge(0,3,w=3.5)
            graph.add_edge(3,4,w=1.5)

            graph.add_edge(0,0,w=10)
            graph.add_edge(1,1,w=10)
            graph.add_edge(2,2,w=10)
            graph.add_edge(3,3,w=10)
            graph.add_edge(4,4,w=10)

            Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=0,init=True)
            while Q:
                Q=GraphUtils.GraphAlgorithm.compute_BF_step(graph=graph,source_id=0,Q=Q)
            for node in graph.nodes():
                print(f"{node} p: {graph.nodes[node]['p']} bf: {graph.nodes[node]['bf']}")
            print()
            GraphUtils.GraphManager.remap_node_predecessor_attr_to_sorted_index(graph=graph)
            for node in graph.nodes():
                print(f"{node} p: {graph.nodes[node]['p']} bf: {graph.nodes[node]['bf']}")


"""
Execute Test
"""
parser=argparse.ArgumentParser()
parser.add_argument("--test",type=int,default=0)
args=parser.parse_args()
test(test_number=args.test)
