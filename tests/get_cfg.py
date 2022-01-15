import yaml
from src.config import *

def get_cfg():
    with open('src/conf/config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        parameters = yaml.load(file, Loader=yaml.FullLoader)

        paths = Paths(
            image_train = parameters['paths']['image_train'].replace("../../../", ""),
            label_train = parameters['paths']['label_train'].replace("../../../", ""),
            image_test = parameters['paths']['image_test'].replace("../../../", ""),
            label_test = parameters['paths']['label_test'].replace("../../../", "")
        )

        img = Image(
            height = parameters["image"]["height"],
            width = parameters["image"]["width"],
            channels = parameters["image"]["channels"]
        )
        pool = MaxPool(
            kernel_size = parameters["maxpool"]["kernel_size"],
            stride = parameters["maxpool"]["stride"],
            padding = parameters["maxpool"]["padding"]
        )
        conv_layers = []
        for layer in parameters["conv_layers"]:
            conv_layers.append(ConvLayer(
                out_channels = layer["out_channels"],
                stride = layer["stride"], 
                padding = layer["padding"], 
                kernel_size =layer["kernel_size"]))
        
        model = Model(
            lr = parameters["model"]["lr"],
            batch_size = parameters["model"]["batch_size"],
            dropout = parameters["model"]["dropout"],
            classes = parameters["model"]["classes"]
        )

        cfg = MNISTConfig(
            image=img, 
            model=model, 
            conv_layers=conv_layers, 
            maxpool=pool,
            paths=paths,
            )
    return cfg