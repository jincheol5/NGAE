import argparse
from utils import DataUtils,TrajUtils

def main(**kwargs):
    mode=kwargs["mode"]
    n_node=kwargs["n_node"]
    graph_type=kwargs["graph_type"]

    # get graph_list
    graph_file_name=f"{mode}_{n_node}"
    graph_list_dict=DataUtils.load_pickle(file_name=graph_file_name,dir_type=f"graph")
    graph_list=graph_list_dict[graph_type]

    # convert to graph_data_list
    graph_data_list=TrajUtils.convert_graph_list(
        graph_list=graph_list,
        graph_type=graph_type
    )

    # save graph_data_list
    file_name=f"traj_{mode}_{n_node}_{graph_type}"
    DataUtils.save_to_pt(
        data=graph_data_list,
        file_name=file_name,
        dir_type=f"trajectory",
        mode=mode
    )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",
        type=str,
        choices=[
            "train",
            "val",
            "test",
        ],
        default=f"ladder"
    )
    parser.add_argument("--n_node",
        type=int,
        choices=[20,50,100],
        default=20
    )
    parser.add_argument("--graph_type",
        type=str,
        choices=[
            "ladder",
            "grid",
            "tree",
            "erdos_renyi",
            "barabasi_albert",
            "community",
            "caveman"
        ],
        default=f"ladder"
    )
    args=parser.parse_args()
    app_config={
        "mode":args.app_num,
        "n_node":args.dataset_name,
        "graph_type":args.sampling
    }
    main(**app_config)