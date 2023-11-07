from TrainerBuilder import TrainBuilder, TrainerConfig
from DataBuilder import DataBuilder, DataConfig
from ModelBuilder import ModelConfig, ModelBuilder
from TrainerClass import Trainer
import timm

import torch
import torch.nn as nn
import torchvision

rotate = torchvision.transforms.RandomRotation(10)

def identity(x):
    return x

commonArgs = dict({
        'dataset' : 'Oxford',
        'task' : 'segmentation',
        'batchSize' : 10,
        'backBone' : 'UNet',
    })

args = dict(commonArgs, **{
        'debugging' : False,
        'amp' : False,
        'debugTest' : False,
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'epochs' : 20,
        'etransforms' : [[rotate, 0,.20]], 
        'equivariant' : True,
        'in_shape' : (3,224,224),
        'n' : 10,
        'learning_rate' : .1,
        'loss' : 'CrossEntropy',
        'endTest' : True
    })

dataArgs = dict(commonArgs, **{
        'split' : 'trainval', 
        'augmented' : False,
        'num_classes' : 3,
    })

testDataArgs = dict(commonArgs, **{
        'split' : 'test',
    })
modelArgs = dict(**{
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'backBone' : 'UNet',
        'in_size' : 768,
        'head' : 'DINO',
        'num_channels' : 3,
        'num_classes' : 3,

    })

trainConfig = TrainerConfig(**args)
dataConfig = DataConfig(**dataArgs)
testConfig = DataConfig(**testDataArgs)

modelConfig = ModelConfig(**modelArgs)

modelBuilder = ModelBuilder(modelConfig)
model = modelBuilder.build()

dataBuilder = DataBuilder(dataConfig)
dataLoader = dataBuilder.makeDataLoader()

testDataBuilder = DataBuilder(testConfig)
testLoader = testDataBuilder.makeDataLoader()

trainer = Trainer(dataConfig, modelConfig, trainConfig, dataConfig, dataLoader, testLoader, model)
trainer.train()   
