import torch
import numpy as np

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

def load_data():
    bs = 32

    X_train = torch.load('data/processed/train_images.pt')
    y_train = torch.load('data/processed/train_labels.pt')

    X_test = torch.load('data/processed/test_images.pt')
    y_test = torch.load('data/processed/test_labels.pt')

    trainloader = torch.utils.data.DataLoader(MNISTDataset(X_train, y_train), shuffle=True, batch_size=bs)
    testloader = torch.utils.data.DataLoader(MNISTDataset(X_test, y_test), shuffle=True, batch_size=bs)

    return trainloader, testloader
