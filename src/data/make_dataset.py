# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np
import glob

def normalize(x):
    return torch.div(torch.subtract(x, x.mean()),x.std())



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_files = glob.glob(input_filepath + '/train*.npz')
    test_files = glob.glob(input_filepath + '/test*.npz')

    train_images = torch.Tensor(np.concatenate([np.load(f)['images'] for f in train_files]))
    train_labels = torch.Tensor(np.concatenate([np.load(f)['labels'] for f in train_files])).type(torch.LongTensor)

    test_images = torch.Tensor(np.concatenate([np.load(f)['images'] for f in test_files]))
    test_labels = torch.Tensor(np.concatenate([np.load(f)['labels'] for f in test_files])).type(torch.LongTensor)


    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, output_filepath + "/train_images.pt")
    torch.save(train_labels, output_filepath + "/train_labels.pt")
    torch.save(test_images, output_filepath + '/test_images.pt')
    torch.save(test_labels, output_filepath + '/test_labels.pt')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
