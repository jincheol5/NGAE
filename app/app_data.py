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
                    num_nodes: 20, 50, 100, 500, 1000
            """
            train_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=100,num_nodes=20)
            val_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20)
            test_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20)
            test_50_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=50)
            test_100_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=100)
            test_500_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=500)
            test_1000_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=1000)

            DataUtils.DataLoader.save_to_pickle(data=train_20_nodes,file_name="train_20_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=val_20_nodes,file_name="val_20_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_20_nodes,file_name="test_20_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_50_nodes,file_name="test_50_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_100_nodes,file_name="test_100_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_500_nodes,file_name="test_500_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_1000_nodes,file_name="test_1000_nodes",dir_type="graph")
        
        case 2:
            """
            App 2.
            Convert graph algo trajectory to PyG Data and save using pickle (num_nodes: 20, 50 ,100)
            """
            train_20_nodes=DataUtils.DataLoader.load_from_pickle(file_name="train_20_nodes",dir_type="graph")
            val_20_nodes=DataUtils.DataLoader.load_from_pickle(file_name="val_20_nodes",dir_type="graph")
            test_20_nodes=DataUtils.DataLoader.load_from_pickle(file_name="test_20_nodes",dir_type="graph")
            test_50_nodes=DataUtils.DataLoader.load_from_pickle(file_name="test_50_nodes",dir_type="graph")
            test_100_nodes=DataUtils.DataLoader.load_from_pickle(file_name="test_100_nodes",dir_type="graph")

            """
            train_20_nodes
            """
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=train_20_nodes,file_name="train_20_nodes")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name="train_20",dir_type="train")

            """
            val_20_nodes
            """
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=val_20_nodes,file_name="val_20_nodes")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name="val_20",dir_type="val")

            """
            test_20_nodes
            """
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=test_20_nodes,file_name="test_20_nodes")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name="test_20",dir_type="test")
        
            """
            test_50_nodes
            """
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=test_50_nodes,file_name="test_50_nodes")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name="test_50",dir_type="test")
            
            """
            test_100_nodes
            """
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=test_100_nodes,file_name="test_100_nodes")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name="test_100",dir_type="test")
        
        case 3:
            """
            App 3.
            Convert graph algo trajectory to PyG Data and save using pickle (num_nodes: 500, 1000)
            """
            graph_list_dict=DataUtils.DataLoader.load_from_pickle(file_name=f"test_{config['test_num_nodes']}_nodes",dir_type="graph")
            graph_list=graph_list_dict[config['graph_type']]
            data_dict=GraphUtils.GraphProcessor.graph_list_to_data_dict(graph_list=graph_list,graph_type=config['graph_type'])
            DataUtils.DataLoader.save_data_dict(data_dict=data_dict,graph_type=config['graph_type'],file_name=f"test_{config['test_num_nodes']}",dir_type="test")


"""
Execute app_data
"""
parser=argparse.ArgumentParser()
parser.add_argument("--app_num",type=int,default=0)
parser.add_argument("--graph_type",type=str,default="ladder")
parser.add_argument("--test_num_nodes",type=int,default=500)
args=parser.parse_args()

config={
    "app_num":args.app_num,
    "graph_type":args.graph_type,
    "test_num_nodes":args.test_num_nodes
}

app_data(config=config)