import argparse
from tqdm import tqdm
from utils import DataUtils,TrainUtils
from torch.utils.data import DataLoader


def main(**kwargs):
    graph_type_list=[
        "ladder",
        "grid",
        "tree",
        "erdos_renyi",
        "barabasi_albert",
        "community",
        "caveman"
    ]
    match kwargs["mode"]:
        case "train":
            ### set train, val dataset
            train_dict={}
            for graph_type in tqdm(graph_type_list,desc=f"Load train dataset..."):
                graph_data_list=DataUtils.load_pt(
                    file_name=f"traj_train_20_{graph_type}",
                    dir_type=f"trajectory",
                    mode="train"
                )
                train_dict[graph_type]=graph_data_list

            val_dict={}
            for graph_type in tqdm(graph_type_list,desc=f"Load val dataset..."):
                graph_data_list=DataUtils.load_pt(
                    file_name=f"traj_val_20_{graph_type}",
                    dir_type=f"trajectory",
                    mode="val"
                )
                val_dict[graph_type]=graph_data_list
            val_data_list=TrainUtils.sampling_k_source_traj_per_graph(
                graph_data_list_dict=val_dict,
                k=10
            )
            val_loader=DataLoader(
                dataset=val_data_list,
                batch_size=kwargs["batch_size"],
                shuffle=False,
                collate_fn=TrainUtils.custom_collate_fn
            )

            # config={
            #     "optimizer":kwargs["optimizer"],
            #     "epoch":kwargs["epoch"],
            #     "lr":kwargs["lr"],
            #     "early_stop":kwargs["early_stop"],
            #     "patience":kwargs["patience"]
            # }

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",
        type=str,
        choices=["train","test"],
        default=f"train"
    )
    parser.add_argument("--model_name",
        type=str,
        choices=["NGAE","CLRS"],
        default=f"NGAE"
    )
    parser.add_argument("--optimizer",
        type=str,
        choices=["adam","sgd"],
        default=f"adam"
    )
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--lr",type=float,default=0.0005)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--early_stop",type=bool,default=True)
    parser.add_argument("--patience",type=int,default=10)
    args=parser.parse_args()
    app_config={
        "mode":args.mode,
        "model_name":args.model_name,
        "optimizer":args.optimizer,
        "epoch":args.epoch,
        "lr":args.lr,
        "batch_size":args.batch_size,
        "early_stop":args.early_stop,
        "patience":args.patience
    }
    main(**app_config)