from re import L
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from src.config import MNISTConfig
import torch
import pytorch_lightning as pl
import torchmetrics

cs = ConfigStore().instance()
cs.store(name='dog_cat_config', node = MNISTConfig)

def compute_conv_dim(dim_size, kernel_size_conv, padding_conv, stride_conv):
    return int((dim_size - kernel_size_conv + 2 * padding_conv) / stride_conv + 1)

class MyAwesomeModel(pl.LightningModule):
    def __init__(self, cfg: MNISTConfig):
        super().__init__()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.layers = nn.ModuleList([])
        hw = []
        self.cfg = cfg
        for idx, conv_layer in enumerate(cfg.conv_layers):
            if idx == 0:
                self.layers.append(
                    nn.Conv2d(
                        in_channels=cfg.image.channels,
                        out_channels=conv_layer.out_channels,
                        kernel_size=conv_layer.kernel_size,
                        stride=conv_layer.stride,
                        padding=conv_layer.padding
                    )
                )
                h = compute_conv_dim(cfg.image.height, conv_layer.kernel_size, conv_layer.padding, conv_layer.stride)
                w = compute_conv_dim(cfg.image.width, conv_layer.kernel_size, conv_layer.padding, conv_layer.stride)
                
            else:
                self.layers.append(
                        nn.Conv2d(
                        in_channels=cfg.conv_layers[idx - 1].out_channels,
                        out_channels=conv_layer.out_channels,
                        kernel_size=conv_layer.kernel_size,
                        stride=conv_layer.stride,
                        padding=conv_layer.padding
                    )
                )
                h = compute_conv_dim(hw[idx - 1][0], conv_layer.kernel_size, conv_layer.padding, conv_layer.stride)
                w = compute_conv_dim(hw[idx - 1][1], conv_layer.kernel_size, conv_layer.padding, conv_layer.stride)
            
            h = compute_conv_dim(h, cfg.maxpool.kernel_size, cfg.maxpool.padding, cfg.maxpool.stride)
            w = compute_conv_dim(w, cfg.maxpool.kernel_size, cfg.maxpool.padding, cfg.maxpool.stride)
            hw.append([h,w])
        

        self.l1_in_features = cfg.conv_layers[-1].out_channels * hw[-1][0] * hw[-1][1]

        self.maxpool = nn.MaxPool2d(cfg.maxpool.kernel_size, cfg.maxpool.stride, padding=cfg.maxpool.padding)

        self.out = nn.Linear(in_features=self.l1_in_features, 
                    out_features=cfg.model.classes,
                    bias=True)

        self.dropout = nn.Dropout(p = cfg.model.dropout)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
            
        for layer in self.layers:
            x = self.maxpool(F.relu(layer(x)))

        x = x.view(-1, self.l1_in_features)
        
        return F.softmax(self.out(x), dim=1)
    
    def loss_func(self, y_hat, y):
        loss = nn.CrossEntropyLoss()
        return loss(y_hat,y)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.model.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss_func(logits,y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss_func(logits,y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        return loss
    