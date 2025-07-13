import os
import random
import numpy as np
import argparse
import torch
from torch_geometric.loader import DataLoader
from ngae import DataUtils,NGAE_BF,NGAE_BFS,ModelTrainer

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
            train single algo
            """
            """
            data loader
            """
            train_data_list=DataUtils.DataLoader.load_from_pickle(file_name="train_20",dir_type="train")
            train_data_loader=DataLoader(dataset=train_data_list,batch_size=1,shuffle=True)
            val_data_loader_dict={}
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20",dir_type="val")
            val_data_loader_dict['all']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_ladder",dir_type="val")
            val_data_loader_dict['ladder']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_grid",dir_type="val")
            val_data_loader_dict['grid']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_tree",dir_type="val")
            val_data_loader_dict['tree']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_erdos_renyi",dir_type="val")
            val_data_loader_dict['erdos_renyi']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_barabasi_albert",dir_type="val")
            val_data_loader_dict['barabasi_albert']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_community",dir_type="val")
            val_data_loader_dict['community']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_caveman",dir_type="val")
            val_data_loader_dict['caveman']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)

            """
            model setting
            """
            match config['task']:
                case 'bfs':
                    model=NGAE_BFS(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
                case 'bf':
                    model=NGAE_BF(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])

            """
            model training
            """
            model=ModelTrainer.train(model=model,train_data_loader=train_data_loader,val_data_loader_dict=val_data_loader_dict,config=config)

        case 2:
            """
            App 2.
            train single algo and save
            """
            """
            data loader
            """
            train_data_list=DataUtils.DataLoader.load_from_pickle(file_name="train_20",dir_type="train")
            train_data_loader=DataLoader(dataset=train_data_list,batch_size=1,shuffle=True)
            val_data_loader_dict={}
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20",dir_type="val")
            val_data_loader_dict['all']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_ladder",dir_type="val")
            val_data_loader_dict['ladder']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_grid",dir_type="val")
            val_data_loader_dict['grid']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_tree",dir_type="val")
            val_data_loader_dict['tree']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_erdos_renyi",dir_type="val")
            val_data_loader_dict['erdos_renyi']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_barabasi_albert",dir_type="val")
            val_data_loader_dict['barabasi_albert']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_community",dir_type="val")
            val_data_loader_dict['community']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)
            val_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_caveman",dir_type="val")
            val_data_loader_dict['caveman']=DataLoader(dataset=val_data_list,batch_size=1,shuffle=True)

            """
            model setting
            """
            match config['task']:
                case 'bfs':
                    model=NGAE_BFS(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])
                case 'bf':
                    model=NGAE_BF(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'])

            """
            model training
            """
            model=ModelTrainer.train(model=model,train_data_loader=train_data_loader,val_data_loader_dict=val_data_loader_dict,config=config)

            """
            save model
            """
            model_name=config['model_name']
            DataUtils.DataLoader.save_model_parameter(model=model,model_name=model_name)
        
        case 3:
            """
            App 3.
            test single algo
            """
            """
            data loader
            """
            test_data_loader_dict={}
            test_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20",dir_type="val")
            test_data_loader_dict['all']=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)
            test_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_ladder",dir_type="val")
            test_data_loader_dict['ladder']=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)
            test_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_grid",dir_type="val")
            test_data_loader_dict['grid']=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)
            test_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_tree",dir_type="val")
            test_data_loader_dict['tree']=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)
            test_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_erdos_renyi",dir_type="val")
            test_data_loader_dict['erdos_renyi']=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)
            test_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_barabasi_albert",dir_type="val")
            test_data_loader_dict['barabasi_albert']=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)
            test_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_community",dir_type="val")
            test_data_loader_dict['community']=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)
            test_data_list=DataUtils.DataLoader.load_from_pickle(file_name="val_20_caveman",dir_type="val")
            test_data_loader_dict['caveman']=DataLoader(dataset=test_data_list,batch_size=1,shuffle=True)

            """
            model test
            """
            for test_graph_type,test_data_loader in val_data_loader_dict.items():
                ModelTrainer.test(model=None,graph_type=test_graph_type,data_loader=test_data_loader,config=config)

"""
Execute app_train
"""
parser=argparse.ArgumentParser()
parser.add_argument("--app_num",type=int,default=1)
parser.add_argument("--task",type=str,default='bfs',required=True)
parser.add_argument("--mode",type=str,default='train')
parser.add_argument("--model_name",type=str,default='NGAE_bfs')
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--latent_dim",type=int,default=32)
parser.add_argument("--optimizer",type=str,default='adam')
parser.add_argument("--lr",type=float,default=0.0005)
parser.add_argument("--epochs",type=int,default=3)
args=parser.parse_args()

config={
    'app_num':args.app_num,
    'task':args.task,
    'mode':args.mode,
    'model_name':args.model_name,
    'seed':args.seed,
    'latent_dim':args.latent_dim,
    'optimizer':args.optimizer,
    'lr':args.lr,
    'epochs':args.epochs
}

app_train(config=config)