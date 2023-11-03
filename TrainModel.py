from TrainerBuilder import TrainBuilder, TrainerConfig
from DataBuilder import DataBuilder, DataConfig
from ModelBuilder import ModelConfig, ModelBuilder
from TrainerClass import Trainer

import torch
import torch.nn as nn

commonArgs = dict({
        'dataset' : 'Oxford',
        'task' : 'category',
        'batchSize' : 1,

    })

args = dict(commonArgs, **{
        'debugging' : False,
        'amp' : True,
        'debugTest' : False,
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    })

dataArgs = dict(commonArgs, **{
        'split' : 'trainval', 
    })

testDataArgs = dict(commonArgs, **{
        'split' : 'test',
    })

trainConfig = TrainerConfig(**args)
dataConfig = DataConfig(**dataArgs)
testConfig = DataConfig(**testDataArgs)

modelConfig = ModelConfig()

modelBuilder = ModelBuilder(modelConfig)
model = modelBuilder.build()

dataBuilder = DataBuilder(dataConfig)
dataLoader = dataBuilder.makeDataLoader()

testDataBuilder = DataBuilder(testConfig)
testLoader = testDataBuilder.makeDataLoader()

trainer = Trainer(dataConfig, modelConfig, trainConfig, dataConfig, dataLoader, testLoader, model)
trainer.train()   
