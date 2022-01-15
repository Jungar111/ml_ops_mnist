import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from hydra.core.config_store import ConfigStore

import torch
from model import MyAwesomeModel
from torch import nn
from torch import optim
from src.data.load_dataset import load_data
import matplotlib.pyplot as plt
import hydra
from src.config import MNISTConfig, register_configs
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

register_configs()

@hydra.main(config_path='../conf', config_name='config')
def main(cfg: MNISTConfig):
    
    model = MyAwesomeModel(cfg)
    
    trainloader, testloader = load_data(cfg, cfg.model.batch_size)

    wandb_logger = WandbLogger(project='mlops_mnist', entity='Jungar', name = 'Initial tests')
    
    trainer = pl.Trainer(logger = wandb_logger, gpus=0, max_epochs=2, log_every_n_steps=100)
    trainer.fit(model, trainloader, testloader)
    trainer.save_checkpoint()

    

            

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()