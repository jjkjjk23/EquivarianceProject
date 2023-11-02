from TrainerBuilder import TrainBuilder, TrainerConfig
from DataBuilder import DataBuilder, DataConfig
from modelBuilder import ModelConfig, ModelBuilder
from TrainerClass import Trainer

import torch
import torch.nn as nn

args = dict({
        'debugging' : False,
        'dataset' : 'Oxford',
        'task' : 'category',
        'amp' : False,
        'batchSize' : 20,
    })



trainConfig = TrainerConfig(**args)
dataConfig = DataConfig(dataset = 'Oxford', task = 'category')
testConfig = DataConfig(dataset = 'Oxford', task = 'category', split = 'test')

modelConfig = ModelConfig()

modelBuilder = ModelBuilder(modelConfig)
model = modelBuilder.build()

dataBuilder = DataBuilder(dataConfig)
dataLoader = dataBuilder.makeDataLoader()

testDataBuilder = DataBuilder(testConfig)
testLoader = testDataBuilder.makeDataLoader()

trainer = Trainer(dataConfig, modelConfig, trainConfig, dataConfig, dataLoader, testLoader, model)
trainer.train()   
