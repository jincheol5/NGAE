import argparse
from ngae import GraphUtils,DataUtils

def app_data(app_number=1):
    match app_number:
        case 1:
            """
            App 1. 
            Generate train, val, test graph list dictionary and save using pickle.
            data info:
                train:
                    graph num of each type: 100 
                    node num: 20

                val:
                    graph num of each type: 3 
                    node num: 20

                test:
                    graph num of each type: 5 
                    node num: 50, 100, 1000
            """
            train_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=100,num_nodes=20)
            val_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20)
            test_20_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20)
            test_50_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=50)
            test_100_nodes=GraphUtils.GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=100)

            DataUtils.DataLoader.save_to_pickle(data=train_20_nodes,file_name="train_20_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=val_20_nodes,file_name="val_20_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_20_nodes,file_name="test_20_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_50_nodes,file_name="test_50_nodes",dir_type="graph")
            DataUtils.DataLoader.save_to_pickle(data=test_100_nodes,file_name="test_100_nodes",dir_type="graph")

"""
Execute app_data
"""
parser=argparse.ArgumentParser()
parser.add_argument("--app",type=int,default=1)
args=parser.parse_args()
app_data(app_number=args.app)