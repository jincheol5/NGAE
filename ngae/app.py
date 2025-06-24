from graph_utils import GraphUtils

graph_dict=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=1,num_nodes=100)

graph=graph_dict['grid'][0]
GraphUtils.GraphVisualizer.visualize_graph(graph=graph)