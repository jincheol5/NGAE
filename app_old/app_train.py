import os
import random
import numpy as np
import argparse
import wandb
import torch
from torch.utils.data import DataLoader
from ngae import DataUtils,ModelTrainUtils,ModelTrainer,NGAE

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
                if config['processor']=='mpnn':
                    wandb.init(project="NGAE",name=f"{config['model']}_{config['aggr']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                else:
                    wandb.init(project="NGAE",name=f"{config['model']}_{config['processor']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                wandb.config.update(config)
            
            """
            data loader
            """
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            train_data_list=[]
            for graph_type in graph_type_list:
                train_data_dict=DataUtils.DataLoader.load_from_pickle(file_name=f"train_20_{graph_type}",dir_type="train")
                if config['random_src']:
                    for _,src_dict in train_data_dict.items():
                        random_src_id=random.randrange(20)
                        train_data_list.append(src_dict[random_src_id])
                else: # all src
                    for _,src_dict in train_data_dict.items():
                        for _,data in src_dict.items():
                            train_data_list.append(data)
            train_data_loader=DataLoader(dataset=train_data_list,batch_size=config['batch_size'],shuffle=True,collate_fn=ModelTrainUtils.batch_collate_fn)

            val_data_list=[]
            for graph_type in graph_type_list:
                val_data_dict=DataUtils.DataLoader.load_from_pickle(file_name=f"val_20_{graph_type}",dir_type="val")
                if config['random_src']:
                    for _,src_dict in val_data_dict.items():
                        random_src_id=random.randrange(20)
                        val_data_list.append(src_dict[random_src_id])
                else: # all src
                    for _,src_dict in val_data_dict.items():
                        for _,data in src_dict.items():
                            val_data_list.append(data)
            val_data_loader=DataLoader(dataset=val_data_list,batch_size=config['batch_size'],shuffle=True,collate_fn=ModelTrainUtils.batch_collate_fn)

            """
            model setting and training
            """
            model=NGAE(node_dim=1,edge_dim=1,latent_dim=config['latent_dim'],processor=config['processor'],aggr=config['aggr'])
            model=ModelTrainer.train(model=model,train_data_loader=train_data_loader,val_data_loader=val_data_loader,config=config)

            """
            save model
            """
            if config['save_model']:
                if config['processor']=='mpnn':
                    model_name=f"ngae_{config['aggr']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                else:
                    model_name=f"ngae_{config['processor']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                DataUtils.DataLoader.save_model_parameter(model=model,model_name=model_name)
        
        case 2:
            """
            App 2.
            test algorithm
            """
            """
            data loader
            """
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            test_data_list=[]
            for graph_type in graph_type_list:
                test_data_dict=DataUtils.DataLoader.load_from_pickle(file_name=f"test_{config['test_num_nodes']}_{graph_type}",dir_type="test")
                for _,src_dict in test_data_dict.items():
                    for _,data in src_dict.items():
                        test_data_list.append(data)
            test_data_loader=DataLoader(dataset=test_data_list,batch_size=config['batch_size'],shuffle=True,collate_fn=ModelTrainUtils.batch_collate_fn)

            """
            model setting and test
            """
            if config['processor']=='mpnn':
                config['model_name']=f"ngae_{config['aggr']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
            else: # gat, gatv2
                config['model_name']=f"ngae_{config['processor']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
            ModelTrainer.test(model=None,graph_type="all",data_loader=test_data_loader,config=config)

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    
    # setting
    parser.add_argument("--algorithm",type=str,default='bfs') # bfs, bf
    parser.add_argument("--processor",type=str,default='mpnn') # mpnn, gat, gatv2
    parser.add_argument("--aggr",type=str,default='max') # max, mean, sum
    parser.add_argument("--optimizer",type=str,default='adam') # adam, sgd
    parser.add_argument("--mode",type=str,default='train') # train, test
    parser.add_argument("--early_stop",type=int,default=1)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--random_src",type=int,default=1)
    

    # train
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--seed",type=int,default=1) # 1, 2, 3
    parser.add_argument("--lr",type=float,default=0.0005) # 0.001, 0,0005
    parser.add_argument("--batch_size",type=int,default=32) # 32, 64
    parser.add_argument("--latent_dim",type=int,default=32)
    
    # 학습 로그 및 저장
    parser.add_argument("--wandb",type=int,default=0)
    parser.add_argument("--save_model",type=int,default=0)

    # 평가
    parser.add_argument("--test_num_nodes",type=int,default=20)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        # setting
        'algorithm':args.algorithm,
        'processor':args.processor,
        'aggr':args.aggr,
        'optimizer':args.optimizer,
        'mode':args.mode,
        'early_stop':args.early_stop,
        'patience':args.patience,
        'random_src':args.random_src,
        # train
        'epochs':args.epochs,
        'seed':args.seed,
        'lr':args.lr,
        'batch_size':args.batch_size,
        'latent_dim':args.latent_dim,
        # 학습 로그 및 저장
        'wandb':args.wandb,
        'save_model':args.save_model,
        # 평가
        'test_num_nodes':args.test_num_nodes
    }
    app_train(config=config)