import argparse
from ngae import GraphUtils,DataUtils

def app_data(config: dict):
    match config["app_num"]:
        case 0:
            print(f"Input application number")
        case 1:
            """
            App 1. 
            Generate train, val, test graph list dictionary and save using pickle.
            data info:
                train:
                    num_graphs (each type): 100 
                    num_nodes: 20

                val:
                    num_graphs (each type): 3 
                    num_nodes: 20

                test:
                    num_graphs (each type): 5 
                    num_nodes: 50, 100, 1000
            """
            # train_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=100,num_nodes=20)
            # val_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20)
            # test_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20)
            # test_50_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=50)
            # test_100_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=100)
            test_1000_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=1000)

            # DataUtils.DataLoader.save_to_pickle(data=train_20_nodes,file_name="train_20_nodes",dir_type="graph")
            # DataUtils.DataLoader.save_to_pickle(data=val_20_nodes,file_name="val_20_nodes",dir_type="graph")
            # DataUtils.DataLoader.save_to_pickle(data=test_20_nodes,file_name="test_20_nodes",dir_type="graph")
            # DataUtils.DataLoader.save_to_pickle(data=test_50_nodes,file_name="test_50_nodes",dir_type="graph")
            # DataUtils.DataLoader.save_to_pickle(data=test_100_nodes,file_name="test_100_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_1000_nodes,file_name="test_1000_nodes",dir_type="graph")
        
        case 2:
            """
            App 2.
            Visualize graph
            """
            file_name=f"{config['task']}_{config['num_nodes']}_nodes"
            graph_list_dict=DataUtils.DataLoader.load_from_pickle(file_name=file_name,dir_type="graph")
            graph=graph_list_dict[config['graph_type']][config['graph_id']]
            GraphUtils.GraphVisualizer.visualize_graph(graph=graph)
        
        case 3:
            """
            App 3.
            Convert graph algo trajectory to PyG Data and save using pickle 
            """
            # train_20_nodes=DataUtils.DataLoader.load_from_pickle(file_name="train_20_nodes",dir_type="graph")
            # val_20_nodes=DataUtils.DataLoader.load_from_pickle(file_name="val_20_nodes",dir_type="graph")
            # test_20_nodes=DataUtils.DataLoader.load_from_pickle(file_name="test_20_nodes",dir_type="graph")
            # test_50_nodes=DataUtils.DataLoader.load_from_pickle(file_name="test_50_nodes",dir_type="graph")
            # test_100_nodes=DataUtils.DataLoader.load_from_pickle(file_name="test_100_nodes",dir_type="graph")
            test_1000_nodes=DataUtils.DataLoader.load_from_pickle(file_name="test_1000_nodes",dir_type="graph")

            # """
            # train_20_nodes
            # """
            # data_list_dict=GraphUtils.GraphProcessor.graph_list_dict_to_PyG_Data_list_dict(graph_list_dict=train_20_nodes,file_name="train_20_nodes")
            # DataUtils.DataLoader.save_data_list_dict(data_list_dict=data_list_dict,file_name="train_20",dir_type="train")

            # """
            # val_20_nodes
            # """
            # data_list_dict=GraphUtils.GraphProcessor.graph_list_dict_to_PyG_Data_list_dict(graph_list_dict=val_20_nodes,file_name="val_20_nodes")
            # DataUtils.DataLoader.save_data_list_dict(data_list_dict=data_list_dict,file_name="val_20",dir_type="val")

            # """
            # test_20_nodes
            # """
            # data_list_dict=GraphUtils.GraphProcessor.graph_list_dict_to_PyG_Data_list_dict(graph_list_dict=test_20_nodes,file_name="test_20_nodes")
            # DataUtils.DataLoader.save_data_list_dict(data_list_dict=data_list_dict,file_name="test_20",dir_type="test")
        
            # """
            # test_50_nodes
            # """
            # data_list_dict=GraphUtils.GraphProcessor.graph_list_dict_to_PyG_Data_list_dict(graph_list_dict=test_50_nodes,file_name="test_50_nodes")
            # DataUtils.DataLoader.save_data_list_dict(data_list_dict=data_list_dict,file_name="test_50",dir_type="test")
            
            # """
            # test_100_nodes
            # """
            # data_list_dict=GraphUtils.GraphProcessor.graph_list_dict_to_PyG_Data_list_dict(graph_list_dict=test_100_nodes,file_name="test_100_nodes")
            # DataUtils.DataLoader.save_data_list_dict(data_list_dict=data_list_dict,file_name="test_100",dir_type="test")

            """
            test_1000_nodes
            """
            data_list=GraphUtils.GraphProcessor.graph_list_to_PyG_Data_list(graph_list=test_1000_nodes['all'],graph_type='all',file_name="test_1000_nodes")
            DataUtils.DataLoader.save_data_list(data_list=data_list,file_name="test_1000",dir_type="test")


"""
Execute app_data
"""
parser=argparse.ArgumentParser()
parser.add_argument("--app_num",type=int,default=0)
parser.add_argument("--task",type=str,default="train")
parser.add_argument("--graph_type",type=str,default="ladder")
parser.add_argument("--graph_id",type=int,default=0)
parser.add_argument("--num_nodes",type=int,default=20)
args=parser.parse_args()

config={
    "app_num":args.app_num,
    "task":args.task,
    "graph_type":args.graph_type,
    "graph_id":args.graph_id,
    "num_nodes":args.num_nodes
}

app_data(config=config)