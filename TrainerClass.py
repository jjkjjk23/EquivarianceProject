import torch
import torch.nn as nn
import time
import logging
from equivariance_regularizer import EquivarianceRegularizer
from pathlib import Path
from TrainerBuilder import TrainBuilder, TrainerConfig
from DataBuilder import DataBuilder, DataConfig
from modelBuilder import ModelConfig, ModelBuilder
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
        self.criterion = nn.CrossEntropyLoss() if self.trainConfig.num_classes > 1 else nn.BCEWithLogitsLoss()
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

    def debugTest(self, global_step):
        if self.trainConfig.debugTest:
            self.endTest(global_step)
        return

    def train(self):
        global_step = 0
        eps=1e-8
        time0=time.time()
        epoch = 0
        self.config()
        self.configOptimizer()
        self.logInit()
        self.wandbInit()
        self.debugTest(global_step)
        for epoch in range(1, self.trainConfig.epochs+1):
            if self.trainConfig.eqweight_scheduler:
                self.eqweightStep()
            self.model.train()
            self.epochTrain(epoch, time0, global_step)
            self.checkpoint()
            self.endTest(global_step)

    def epochTrain(self, epoch, time0, global_step):

        with tqdm(
                total=len(self.train_loader),
                desc=f'Epoch {epoch}/{self.trainConfig.epochs}',
                unit='img'
                ) as pbar:
            epochStep = 0
            epochTime = time.time()
            for batch in self.train_loader:
                images, true_masks = batch
                with torch.autocast(self.trainConfig.device.type if self.trainConfig.device.type != 'mps' else 'cpu', enabled=self.trainConfig.amp):
                    self.model.to(dtype = images.dtype, device = images.device)
                    masks_pred = self.model(images)
                    if not self.trainConfig.debugging:
                        self.basicTrainLog(epochTime, time0, global_step, epoch)
                        self.logImages(global_step, images, true_masks, masks_pred)
                    del images
                    loss = self.loss(true_masks, masks_pred)
                self.step(loss)
                pbar.update()
                if not self.trainConfig.debugging:
                    self.lossLog(loss, true_masks, masks_pred,pbar)
                del loss
                del true_masks
                del masks_pred
                self.equivarianceStep()
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

    def logImages(self,global_step, images, true_masks, masks_pred):
        if global_step % (len(self.train_loader)//5)==0:
            if self.trainConfig.task == 'category':
                self.labelledImageLog(images, true_masks, masks_pred)
            elif self.trainConfig.task == 'segmentation':
                self.basicImageLog(images, true_masks, masks_pred)
        return

    def loss(self, true_masks, masks_pred):
        loss = self.criterion(masks_pred, true_masks)
        if self.trainConfig.task=='segmentation':
            loss+=self.diceLoss(masks_pred, true_masks)
        return loss

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
                self.experiment.log({'Equivariance error' : equivarianceError})
        return


    def lossLog(self, loss,true_masks,masks_pred, pbar):
        if self.trainConfig.task=='segmentation':
            self.logDice(true_masks, masks_pred)
        if self.trainConfig.task == 'category':
            self.logAccuracy(true_masks, masks_pred)
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

    def endTest(self, global_step):
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

                    self.logImages(global_step, images, true_masks, masks_pred)
                    del images

                    testLoss += float(self.loss(true_masks, masks_pred))
                    if self.trainConfig.task == 'segmentation':
                        for i in range(len(dice)):
                            dice[i]+=self.calcDice(true_masks, masks_pred)
                    elif self.trainConfig.task == 'category':
                        accuracy += self.calcAccuracy(true_masks, masks_pred)

                    del true_masks
                    del masks_pred

                    equivarianceError += float(self.equivarianceFunction())
                    step+=1
                    global_step+=1
                    pbar.update()

        self.experiment.log({
            'Test Equivariance Error' : equivarianceError,
            })
        if self.trainConfig.task == 'segmentation':
            self.experiment.log({'Test Dice' : dice})

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
