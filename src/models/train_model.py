import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
from model import MyAwesomeModel
from torch import nn
from torch import optim
from src.data.load_dataset import load_data
import matplotlib.pyplot as plt

def main():
    model = MyAwesomeModel()
    trainloader, testloader = load_data()
    training_loss = []
    validation_loss = []
    ep = []
    acc_val = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    bs = 32
    epochs = 5
    for e in range(epochs):
        running_loss = 0
        acc = 0
        val_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            with torch.no_grad():
                for images, labels in testloader:
                    out = model(images.float())
                    predicted = torch.argmax(out, 1)
                    acc += (predicted == labels).sum()
                    
                    val_loss += criterion(out, labels).item()
            
            accuracy = acc/(len(testloader) * bs)
            
            ep.append(e)
            training_loss.append(running_loss/(len(trainloader)*bs))
            validation_loss.append(val_loss/(len(testloader)*bs))
            acc_val.append(accuracy.item()*100)
            print('-'* 10, ' EPOCH: ', str(e) + ' ', '-'*10)
    
    torch.save(model.state_dict(), 'MNIST/models/MNISTmodel.pth')
    plt.plot(ep, training_loss)
    plt.plot(ep, validation_loss)
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("reports/figures/losscurve.png")
    plt.close()

    plt.plot(acc_val)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("reports/figures/accuracy.png")

            

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()