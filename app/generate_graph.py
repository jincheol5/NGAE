from utils import GraphGenerator,DataUtils

def main():
    """
    graph info:
        train:
            n_graph (each type): 100 
            n_node: 20

        val:
            n_graph (each type): 5 
            n_node: 20

        test:
            n_graph (each type): 5 
            n_node: 20, 50, 100
    """
    train_20=GraphGenerator.generate_7_type_graphs(n_graph=100,n_node=20)
    val_20=GraphGenerator.generate_7_type_graphs(n_graph=5,n_node=20)
    test_20=GraphGenerator.generate_7_type_graphs(n_graph=5,n_node=20)
    test_50=GraphGenerator.generate_7_type_graphs(n_graph=5,n_node=50)
    test_100=GraphGenerator.generate_7_type_graphs(n_graph=5,n_node=100)

    DataUtils.save_to_pickle(data=train_20,file_name=f"graph_train_20",dir_type=f"graph")
    DataUtils.save_to_pickle(data=val_20,file_name=f"graph_val_20",dir_type=f"graph")
    DataUtils.save_to_pickle(data=test_20,file_name=f"graph_test_20",dir_type=f"graph")
    DataUtils.save_to_pickle(data=test_50,file_name=f"graph_test_50",dir_type=f"graph")
    DataUtils.save_to_pickle(data=test_100,file_name=f"graph_test_100",dir_type=f"graph")

if __name__=="__main__":
    """
    Execute app
    """
    main()