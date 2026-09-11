import argparse
from ngae import GraphUtils,DataUtils

def app_data(config:dict):
    match config["app_num"]:
        case 1:
            """
            App 1. 
            Generate train, val, test graph list dictionary and save using pickle.
            data info:
                train:
                    num_graphs (each type): 100 
                    num_nodes: 20

                val:
                    num_graphs (each type): 5 
                    num_nodes: 20

                test:
                    num_graphs (each type): 5 
                    num_nodes: 20, 50, 100, 500, 1000
            """
            train_20=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=100,num_nodes=20)
            val_20=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20)
            test_20=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20)
            test_50=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=50)
            test_100=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=100)
            test_500=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=500)
            test_1000=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=1000)

            DataUtils.DataLoader.save_to_pickle(data=train_20,file_name="train_20",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=val_20,file_name="val_20",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_20,file_name="test_20",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_50,file_name="test_50",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_100,file_name="test_100",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_500,file_name="test_500",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_1000,file_name="test_1000",dir_type="graph")

        case 2:
            """
            App 2.
            Convert graph algo trajectory to PyG Data and save using pickle (num_nodes: 20, 50, 100)
            """
            train_20=DataUtils.DataLoader.load_from_pickle("train_20","graph")
            val_20=DataUtils.DataLoader.load_from_pickle("val_20","graph")
            test_20=DataUtils.DataLoader.load_from_pickle("test_20","graph")
            test_50=DataUtils.DataLoader.load_from_pickle("test_50","graph")
            test_100=DataUtils.DataLoader.load_from_pickle("test_100","graph")

            # train_20_nodes
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=train_20,file_name="train_20")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name=f"train_20",dir_type="train")

            # val_20_nodes
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=val_20,file_name="val_20")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name=f"val_20",dir_type="val")

            # test_20_nodes
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=test_20,file_name="test_20")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name=f"test_20",dir_type="test")

            # test_50_nodes
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=test_50,file_name="test_50")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name=f"test_50",dir_type="test")

            # test_100_nodes
            all_data_dict=GraphUtils.GraphProcessor.graph_list_dict_to_all_data_dict(graph_list_dict=test_100,file_name="test_100")
            DataUtils.DataLoader.save_all_data_dict(all_data_dict=all_data_dict,file_name=f"test_100",dir_type="test")
        
        case 3:
            """
            App 3.
            Convert graph algo trajectory to PyG Data and save using pickle (num_nodes: 500, 1000)
            """
            graph_list_dict=DataUtils.DataLoader.load_from_pickle(f"test_{config['test_num_nodes']}","graph")
            graph_list=graph_list_dict[config['graph_type']]
            data_dict=GraphUtils.GraphProcessor.graph_list_to_data_dict(graph_list=graph_list,graph_type=config['graph_type'])
            DataUtils.DataLoader.save_data_dict(data_dict=data_dict,file_name=f"test_{config['test_num_nodes']}_{config['graph_type']}",dir_type="test")

if __name__=="__main__":
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