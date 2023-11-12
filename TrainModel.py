from TrainerBuilder import TrainBuilder, TrainerConfig
from DataBuilder import DataBuilder, DataConfig
from ModelBuilder import ModelConfig, ModelBuilder
from TrainerClass import Trainer
import timm

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
randomResizedCrop = v2.RandomResizedCrop((518, ))
etransforms = [
            [identity, rotate, 0, .00005],
            [identity, horizFlip, 0, .00005],
            [identity, perspective, 0, .00005],
            [identity, color, 0, .00005],
            [identity, randomResizedCrop, 0.01, .00005]
               ]





commonArgs = dict({
        'dataset' : 'Oxford',
        'task' : 'category',
        'batchSize' : 10, 
        'backBone' : 'DINO',
        'in_shape' : (3, 336, 336),
    })

args = dict(commonArgs, **{
        'debugging' : False,
        'amp' : False,
        'debugTest' : False,
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'epochs' : 25,
        'etransforms' : etransforms, 
        'equivariant' : True,
        'n' : 10,
        'learning_rate' : .1,
        'loss' : 'myLoss',
        'endTest' : True,
        'wandb_project' : "EquivarianceProject",
        'numFuncs' : 5,
        'bounds' : [-1,1],
    })

dataArgs = dict(commonArgs, **{
        'split' : 'trainval', 
        'augmented' : False,
        'num_classes' : 37,
    })

testDataArgs = dict(commonArgs, **{
        'split' : 'test',
    })
modelArgs = dict(**{
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'backBone' : 'DINO',
        'in_size' : 1024,
        'head' : 'DINO',
        'num_channels' : 3,
        'num_classes' : 37,
        'size' : "large",

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
for _ in range(10):
    model = modelBuilder.build()
    trainer = Trainer(dataConfig, modelConfig, trainConfig, dataConfig, dataLoader, testLoader, model)
    trainer.train()   
