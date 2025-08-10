import os
import random
import numpy as np
import argparse
import wandb
import torch
from torch_geometric.loader import DataLoader
from ngae import DataUtils,ModelTrainer,NGAE

def app_train(config: dict):
    """
    seed setting
    """
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    os.environ["PYTHONHASHSEED"]=str(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic=True 
    torch.backends.cudnn.benchmark=False

    match config['app_num']:
        case 1:
            """
            App 1.
            train algorithm trajectory
            """
            if config['wandb']:
                wandb.init(project="NGAE",name=f"{config['algorithm']}")
                wandb.config.update(config)

            """
            data loader
            """
            train_data_list=[]
            train_data_dict=DataUtils.DataLoader.load_from_pickle(file_name="train_20_all",dir_type="train")
            if config['random_src']:
                for _,src_dict in train_data_dict.items():
                    random_src_id=random.randrange(20)
                    train_data_list.append(src_dict[random_src_id])
            else: # all src
                for _,src_dict in train_data_dict.items():
                    for _,data in src_dict.items():
                        train_data_list.append(data)
            train_data_loader=DataLoader(dataset=train_data_list,batch_size=1,shuffle=True)

            val_data_list=[]
            val_data_dict=DataUtils.DataLoader.load_from_pickle(file_name="val_20_all",dir_type="val")
            if config['random_src']:
                for _,src_dict in val_data_dict.items():
                    random_src_id=random.randrange(20)
                    val_data_list.append(src_dict[random_src_id])
            else: # all src
                for _,src_dict in val_data_dict.items():
                    for _,data in src_dict.items():
                        val_data_list.append(data)
            val_data_loader=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)

            """
            model setting
            """
            model=NGAE(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'],algorithm=config['algorithm'],processor=config['processor'],aggr=config['aggr'])

            """
            model training
            """
            model=ModelTrainer.train(model=model,train_data_loader=train_data_loader,val_data_loader=val_data_loader,config=config)

            """
            save model
            """
            if config['save_model']:
                model_name=config['model_name']
                DataUtils.DataLoader.save_model_parameter(model=model,model_name=model_name)

        case 2:
            """
            App 2.
            test algorithm
            """
            """
            data loader
            """
            test_data_loader_dict={}
            graph_type_list=['all','ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            for graph_type in graph_type_list:
                test_data_list=[]
                test_data_dict=DataUtils.DataLoader.load_from_pickle(file_name=f"test_{config['test_num_nodes']}_{graph_type}",dir_type="test")
                if config['random_src']:
                    for _,src_dict in test_data_dict.items():
                        random_src_id=random.randrange(20)
                        test_data_list.append(src_dict[random_src_id])
                else: # all src
                    for _,src_dict in test_data_dict.items():
                        for _,data in src_dict.items():
                            test_data_list.append(data)
                test_data_loader=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)
                test_data_loader_dict[graph_type]=test_data_loader

            """
            model test
            """
            for test_graph_type,test_data_loader in test_data_loader_dict.items():
                ModelTrainer.test(model=None,graph_type=test_graph_type,data_loader=test_data_loader,config=config)


"""
Execute app_train
"""
parser=argparse.ArgumentParser()
# app 관련
parser.add_argument("--app_num",type=int,default=1)
parser.add_argument("--seed",type=int,default=42)
# train 관련
parser.add_argument("--lr",type=float,default=0.0005)
parser.add_argument("--optimizer",type=str,default='adam')
parser.add_argument("--epochs",type=int,default=1)
parser.add_argument("--patience",type=int,default=1)
parser.add_argument("--random_src",type=int,default=1)
# model 설정 관련
parser.add_argument("--algorithm",type=str,default='bfs')
parser.add_argument("--mode",type=str,default='train')
parser.add_argument("--processor",type=str,default='mpnn')
parser.add_argument("--aggr",type=str,default='max')
parser.add_argument("--latent_dim",type=int,default=32)
# test 및 검증 관련
parser.add_argument("--test_num_nodes",type=int,default=20)
parser.add_argument("--wandb",type=int,default=0)
parser.add_argument("--save_model",type=int,default=0)
parser.add_argument("--model_name",type=str,default='CLRS')
args=parser.parse_args()

config={
    'app_num':args.app_num,
    'seed':args.seed,
    'lr':args.lr,
    'optimizer':args.optimizer,
    'epochs':args.epochs,
    'patience':args.patience,
    'random_src':args.random_src,
    'algorithm':args.algorithm,
    'mode':args.mode,
    'processor':args.processor,
    'aggr':args.aggr,
    'latent_dim':args.latent_dim,
    'model_name':args.model_name,
    'test_num_nodes':args.test_num_nodes,
    'wandb':args.wandb,
    'save_model':args.save_model
}
app_train(config=config)