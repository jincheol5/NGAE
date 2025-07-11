import torch
from tqdm import tqdm

class ModelTrainer:
    @staticmethod
    def train(model,train_data_loader,val_data_loader,config: dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])

        for epoch in range(config['epochs']):
            model.train()
            for batch in train_data_loader:
                batch=batch.to(device)
                h_0=torch.zeros((batch.num_nodes,config.latent_dim),dtype=torch.float32)
                h_0=h_0.to(device)

                match config['task']:
                    case 'bfs':
                        algo_trajectory=batch.bfs
                    case 'bf':
                        algo_trajectory=batch.bf

                output=model(algo_trajectory=algo_trajectory,h_0=h_0,edge_index=batch.edge_index,edge_attr=batch.edge_attr,task='train')