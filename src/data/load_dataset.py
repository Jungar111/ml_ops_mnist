import torch
import numpy as np
import hydra
from src.config import MNISTConfig
from hydra.core.config_store import ConfigStore


cs = ConfigStore().instance()
cs.store(name='mnist_config', node = MNISTConfig)

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X = self.X.reshape(X.size(0), 1, X.size(1),X.size(2))
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]

def concat_multiple(sets):
    new = np.concatenate((sets[0],sets[1]))
    for i in range(2, len(sets)):
        new = np.concatenate((new, sets[i]))
    
    return new

@hydra.main(config_path='../conf', config_name='config')
def load_data(cfg: MNISTConfig):
    bs = cfg.model.batch_size

    X_train = torch.load(cfg.paths.image_train)
    y_train = torch.load(cfg.paths.label_train)

    X_test = torch.load(cfg.paths.image_test)
    y_test = torch.load(cfg.paths.image_test)

    trainloader = torch.utils.data.DataLoader(MNISTDataset(X_train, y_train), shuffle=True, batch_size=bs)
    testloader = torch.utils.data.DataLoader(MNISTDataset(X_test, y_test), shuffle=True, batch_size=bs)
    
    hydra._internal.hydra.GlobalHydra().clear() 

    return trainloader, testloader
