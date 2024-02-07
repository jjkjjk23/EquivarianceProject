from TrainerBuilder import TrainBuilder, TrainerConfig
from DataBuilder import DataBuilder, DataConfig
from ModelBuilder import ModelConfig, ModelBuilder
from TrainerClass import Trainer
from etrainerfunctions import shift
import timm
import random

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import numpy as np


def identity(x):
    return x

rotate = torchvision.transforms.RandomRotation(10)
horizFlip = torchvision.transforms.RandomHorizontalFlip(1)
perspective = v2.RandomPerspective()
color = v2.ColorJitter()

commonArgs = dict({
        'dataset': 'Oxford',
        'task': 'segmentation',
        'batchSize': 1,
        'backBone': 'UNet',
        'in_shape': (3, 375, 500),
    })

randomResizedCrop = v2.RandomResizedCrop(commonArgs['in_shape'][1:])
eqweight = .01

def random_shift(tensor):
    shiftnum = random.randint(-50,50)
    return shift(tensor, shiftnum, -1)

catEtransforms = [
            [identity, rotate, 0, eqweight],
            [identity, horizFlip, 0, eqweight],
            [identity, perspective, 0, eqweight],
            [identity, color, 0, eqweight],
            [identity, randomResizedCrop, 0.01, eqweight]
               ]

etransforms = [[random_shift, 0, eqweight]]

args = dict(commonArgs, **{
        'debugging': True,
        'amp': False,
        'debugTest': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'epochs': 25,
        'etransforms': etransforms,
        'equivariant': True,
        'n': 1,
        'learning_rate': .001,
        'loss': 'CrossEntropy', #myLoss: for categorical, 'BCE' and 'CrossEntropy' for                             segmentation
        'endTest': False,
        'wandb_project': "EquivarianceProject",
        'numFuncs': 1,
        'bounds': [-1,1],
        'save_checkpoint': True,
    })

dataArgs = dict(commonArgs, **{
        'split': 'trainval',
        'augmented': False,
        'num_classes': 3,
    })

testDataArgs = dict(commonArgs, **{
        'split': 'test',
    })
modelArgs = dict(**{
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'backBone': 'UNet',
        'in_size': 1024,
        'head': 'custom',
        'num_channels': 3,
        'num_classes': 3,
        'size': "large",

    })

trainConfig = TrainerConfig(**args)
dataConfig = DataConfig(**dataArgs)
testConfig = DataConfig(**testDataArgs)

modelConfig = ModelConfig(**modelArgs)

modelBuilder = ModelBuilder(modelConfig)

dataBuilder = DataBuilder(dataConfig)
dataLoader = dataBuilder.makeDataLoader()

testDataBuilder = DataBuilder(testConfig)
testLoader = testDataBuilder.makeDataLoader()

modelPath = "../checkpoints/magic-river-135/checkpoint_epoch3.pth"
#model = torch.compile(modelBuilder.build().to(args['device']))
model = modelBuilder.build().to(args['device'])
#state_dict = torch.load(modelPath, map_location=args['device'])
#model.load_state_dict(state_dict)
trainer = Trainer(dataConfig, modelConfig, trainConfig, dataConfig, dataLoader, testLoader, model)
trainer.train()   
