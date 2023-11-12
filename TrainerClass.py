import torch
import torch.nn as nn
import time
import logging
from equivariance_regularizer_test import EquivarianceRegularizer
from pathlib import Path
from TrainerBuilder import TrainBuilder, TrainerConfig
from DataBuilder import DataBuilder, DataConfig
from ModelBuilder import ModelConfig, ModelBuilder
from tqdm import tqdm

import wandb

class Trainer(TrainBuilder):
    def __init__(
            self, 
            *args
            ):
        super().__init__(*args)

    def config(self): 
        if self.scheduler == 'OnPlateau':
            self.plateauConfig()
        if self.scheduler == 'cyclic':
            self.cyclicConfig()
        if self.trainConfig.gradientScaling:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.trainConfig.amp)
        if self.trainConfig.loss == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss() if self.trainConfig.num_classes > 1 else nn.BCEWithLogitsLoss()
        elif self.trainConfig.loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.trainConfig.loss == 'myLoss':
            self.criterion = self.myLoss
        self.equivarianceFunction = self.configEquivariance()
        self.configOptimizer()
        return

    def logInit(self):
        logging.info(f'''Starting training:
            Epochs:          {self.trainConfig.epochs}
            Batch size:      {self.trainConfig.batchSize}
            Learning rate:   {self.trainConfig.learning_rate}
            Checkpoints:     {self.trainConfig.save_checkpoint}
            Device:          {self.trainConfig.device.type}
            Mixed Precision: {self.trainConfig.amp}
            transforms: {self.trainConfig.etransforms}
            equivariant: {self.trainConfig.equivariant}
            Equivariance weight: {self.trainConfig.eqweight}
            n: {self.trainConfig.n}
            augmented: {self.testConfig.augmented}
        ''')
        return
    def configOptimizer(self):
        if self.trainConfig.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                    self.model.parameters(), 
                    lr=self.trainConfig.learning_rate, 
                    momentum=0.9)
        elif self.trainConfig.optimizer == 'RMS':
            self.optimizer = torch.optim.RMSProp(
                    self.model.parameters(),
                    lr = self.trainConfig.learning_rate)
        elif self.trainConfig.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                    self.model.parameters(), 
                    lr = self.trainConfig.learning_rate)
    def wandbInit(self):
        if self.trainConfig.debugging:
            self.debugWandb()
            return None
        else:
            return self.stdWandb()

    def debugTest(self):
        if self.trainConfig.debugTest:
            self.endTest()
        return

    def train(self):
        self.global_step = 0
        eps=1e-8
        time0=time.time()
        epoch = 0
        self.config()
        self.configOptimizer()
        self.logInit()
        self.wandbInit()
        self.debugTest()
        for epoch in range(1, self.trainConfig.epochs+1):
            if self.trainConfig.eqweight_scheduler:
                self.eqweightStep()
            self.model.train()
            self.epochTrain(epoch, time0)
            self.checkpoint()
            if self.trainConfig.endTest:
                self.endTest()

    def epochTrain(self, epoch, time0):

        with tqdm(
                total=len(self.train_loader),
                desc=f'Epoch {epoch}/{self.trainConfig.epochs}',
                unit='img'
                ) as pbar:
            epochStep = 0
            epochTime = time.time()
            for batch in self.train_loader:
                with torch.autocast(
                        self.trainConfig.device.type,
                        enabled=self.trainConfig.amp):
                    images, true_masks = batch
                    self.model.to(dtype = images.dtype, device = images.device)
                    masks_pred = self.model(images)
                    if not self.trainConfig.debugging:
                        self.basicTrainLog(epochTime, time0,epoch)
                        if self.global_step % (len(self.train_loader)//4) ==0:
                            self.logImages(images, masks_pred, true_masks)
                    del images
                    loss = self.loss(masks_pred, true_masks)
                self.step(loss)
                pbar.update()
                if not self.trainConfig.debugging:
                    self.lossLog(loss, masks_pred, true_masks,pbar)
                del loss
                del true_masks
                del masks_pred
                self.equivarianceStep()
                epochStep+=1
                self.global_step+=1
#                if self.global_step>=200:
#                    break
        return
    #Move this into configData
    def processImages(self, images, true_masks):
        if self.trainConfig.dataset == 'Oxford':
            if self.trainConfig.task == 'category':
                self.oxfordSegProcessing(true_masks)
                return
            else:
                self.oxfordCatProcessing(true_masks)
                return
        elif self.trainConfig.dataset == 'HeLa':
            self.heLaProcessing(images, true_masks)
            return

    def logImages(self, images, masks_pred, true_masks):
        if self.trainConfig.task == 'category':
            self.labelledImageLog(images, masks_pred, true_masks)
        elif self.trainConfig.task == 'segmentation':
            self.basicImageLog(images, masks_pred, true_masks)
        return

    def loss(self, masks_pred, true_masks):
        loss = self.criterion(masks_pred, true_masks)
        if self.trainConfig.task=='segmentation':
            loss+=self.diceLoss(masks_pred, true_masks)
        return loss

    def myLoss(self, masks_pred, true_masks):
        masks_pred = -nn.functional.log_softmax(masks_pred, dim = -1)
        masks_pred, _ = torch.max(masks_pred*true_masks, dim = -1)
        return torch.mean(masks_pred)

    def accuracy(self):
        return

    def step(self, term):
        if self.trainConfig.gradientScaling:
            self.UNetStep(term)
        else:
            self.baseStep(term)
        return

    def equivarianceStep(self):
        if self.trainConfig.equivariant:
            equivarianceError = self.equivarianceFunction()
            self.step(equivarianceError)
            if not self.trainConfig.debugging:
                self.experiment.log(
                    {'Equivariance error' : equivarianceError/self.trainConfig.etransforms[-1][-1]})
        return


    def lossLog(self, loss,masks_pred, true_masks, pbar):
        if self.trainConfig.task=='segmentation':
            self.logDice(masks_pred, true_masks)
        if self.trainConfig.task == 'category':
            self.logAccuracy(masks_pred, true_masks)
        self.experiment.log({'Loss' : loss})
        pbar.set_postfix(**{'Loss' : loss.item()}) 
        return

    def checkpoint(self):
        if self.trainConfig.save_checkpoint and not self.trainConfig.debugging:
            dir_checkpoint = Path(f'../checkpoints/{str(experiment.name)}/')
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = self.model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            
        return

    def endTest(self):
        equivarianceError = 0
        testLoss = 0
        accuracy = 0
        dice = [0, 0, 0]
        step = 0
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                    total=len(self.testLoader),
                    unit='img'
                    ) as pbar:
                for batch in self.testLoader:
                    images, true_masks = batch
                    self.model.to(dtype = images.dtype, device = images.device)
                    masks_pred = self.model(images)
            
                    if step % len(self.testLoader)//5 == 0:
                        self.logImages( images, masks_pred, true_masks)
                    equivarianceError += float(self.equivarianceFunction(images))
                    del images

                    testLoss += float(self.loss(masks_pred, true_masks))
                    if self.trainConfig.task == 'segmentation':
                        new_dice = self.calcDice(masks_pred, true_masks)
                        for i in range(len(dice)):
                            dice[i]+=float(new_dice[i])
                    elif self.trainConfig.task == 'category':
                        accuracy += self.calcAccuracy(masks_pred, true_masks)
                    del true_masks
                    del masks_pred
                    step+=1
                    self.global_step+=1
                    pbar.update()
                    pbar.set_postfix({'Test Accuracy' : accuracy/step})
        dice = [i/step for i in dice]
        accuracy = accuracy/step
        testLoss = testLoss/step
        equivarianceError = equivarianceError/step
        self.experiment.log({
            'Test Equivariance Error' : equivarianceError/self.trainConfig.etransforms[-1][-1],
            'Test Loss' : testLoss,
            })

        if self.trainConfig.task == 'segmentation':
            self.experiment.log({f'Test Dice {i}' : dice[i] for i in range(len(dice))})

        if self.trainConfig.task == 'category':
            self.experiment.log({'Test Accuracy' : accuracy})

        self.model.train()

        return

    def epochBreak(self):
        return

    def makeModel(self):
        return

    def makeTrainLoader(self):
        return

    def makeTestLoader(self):
        return
