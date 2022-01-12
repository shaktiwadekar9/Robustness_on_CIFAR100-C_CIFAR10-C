import torch
from torch import nn
import models
import models.resnet as resnet
from models.mobilenet.mobilenetv2 import MobileNetV2Wrapper



# Main function
def get_model(cfg):
    """ Return the model which matches to the name in config.yml file
    """
    # Load model
    if cfg['model']['name']=='resnet50':
        model = resnet.resnet50(cfg['model']['n_cls'])
        print("Resnet50 model created!")
    elif cfg['model']['name']=='mobilenetv2':
        model = MobileNetV2Wrapper(cfg['model']['n_cls'])
        print("MobileNetV2 baseline model created!")
    else:
        raise NotImplementedError(" Given model not implemented")


    return model

