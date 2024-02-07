import torch
import torch.nn as nn
import time
import logging
from equivariance_regularizer_test import EquivarianceRegularizer
from pathlib import Path
from DataBuilder import DataBuilder, DataConfig
from ModelBuilder import ModelConfig, ModelBuilder
from tqdm import tqdm
from etrainerfunctions import dicescore

import wandb

@torch.compile
def trainingStep(trainer, batch, epoch, epochStep, time0, pbar):
    with torch.autocast(
            trainer.trainConfig.device.type,
            enabled=trainer.trainConfig.amp):
        images, true_masks = batch
        trainer.model.to(dtype = images.dtype, device = images.device)
        masks_pred = trainer.model(images)
        if not trainer.trainConfig.debugging:
            trainer.basicTrainLog(epochTime, time0,epoch)
            if trainer.global_step % (len(trainer.train_loader)//4) ==0:
                trainer.logImages(images, masks_pred, true_masks)
        del images
        loss = trainer.loss(masks_pred, true_masks)
    trainer.step(loss)
    pbar.update()
    if not trainer.trainConfig.debugging:
        trainer.lossLog(loss, masks_pred, true_masks,pbar)
    del loss
    del true_masks
    del masks_pred
    trainer.equivarianceStep()
    epochStep+=1
    trainer.global_step+=1
#                if self.global_step>=200:
#                    break
@torch.compile
def epochTrain(trainer, epoch, time0):
    with tqdm(
            total=len(trainer.train_loader),
            desc=f'Epoch {epoch}/{trainer.trainConfig.epochs}',
            unit='img'
            ) as pbar:
        epochStep = 0
        epochTime = time.time()
        for batch in trainer.train_loader:
            trainingStep(trainer, batch, epoch, epochStep, time0, pbar)
    return


class TrainerConfig:
    def __init__(
            self,
            debugging=False,
            etransforms=None,
            equivariant=False,
            eqweight=None,
            n=0,
            class_weights=None,
            save_checkpoint=False,
            min_lr=0,
            max_lr=.01,
            wandb_project=None,
            eqweight_scheduler=False,
            task='category',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            epochs=1,
            amp=True,
            batchSize=1,
            num_classes=1,
            gradient_clipping=1.0,
            dataset='HeLa',
            scheduler=None,
            optimizer='SGD',
            gradientScaling=False,
            learning_rate=.1,
            debugTest=True,
            in_shape=(3,224,224),
            loss='myLoss',
            endTest=False,
            num_funcs=1,
            bounds=[-1,1],
            **kwargs,
            ):
        self.debugging = debugging
        self.etransforms = etransforms
        self.equivariant = equivariant
        self.eqweight = eqweight
        self.n = n
        self.class_weights = None
        self.save_checkpoint = save_checkpoint
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wandb_project = wandb_project
        self.eqweight_scheduler = eqweight_scheduler
        self.task = task
        self.device = device
        self.epochs = epochs
        self.amp = amp
        self.batchSize = batchSize
        self.num_classes = num_classes
        self.gradient_clipping = gradient_clipping
        self.dataset = 'HeLa'
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.gradientScaling = gradientScaling
        self.learning_rate = learning_rate
        self.debugTest = debugTest
        self.in_shape = in_shape
        self.loss = loss
        self.endTest = endTest
        self.num_funcs = num_funcs
        self.bounds = bounds

class Trainer():
    def __init__(
            self,
            dataConfig,
            modelConfig,
            trainConfig,
            testConfig,
            train_loader,
            testLoader,
            model,
            *args,
            ):
        self.scheduler = trainConfig.scheduler
        self.experiment = None
        self.grad_scaler = None
        self.criterion = None
        self.trainConfig = trainConfig
        self.eqweight = trainConfig.eqweight
        self.model = model
        self.modelConfig = modelConfig
        self.dataConfig = dataConfig
        self.testConfig = testConfig
        self.train_loader = train_loader
        self.testLoader = testLoader

        self.global_step = 0

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
            epochTrain(self, epoch, time0)
            self.checkpoint(epoch)
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
                trainingStep(self, batch, epoch, epochStep, time0, pbar)
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

    def checkpoint(self, epoch):
        if self.trainConfig.save_checkpoint and not self.trainConfig.debugging:
            dir_checkpoint = Path(f'../checkpoints/{str(self.experiment.name)}/')
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

#Helper functions begin here

    def debugWandb(self):
        print('This is running in debugging mode and is not logging with wandb')
        return
    def stdWandb(self):
        self.experiment = wandb.init(project=self.trainConfig.wandb_project, resume='allow', anonymous='must')
        #min and max lr?
        self.experiment.config.update(
            dict(epochs=self.trainConfig.epochs,
                 batchSize=self.trainConfig.batchSize,
                 learning_rate=self.trainConfig.learning_rate,
                 save_checkpoint=self.trainConfig.save_checkpoint,
                 amp=self.trainConfig.amp,
                 etransforms = self.trainConfig.etransforms,
                 equivariant = self.trainConfig.equivariant,
                 eqweight=self.trainConfig.eqweight,
                 n=self.trainConfig.n,
                 class_weights=self.trainConfig.class_weights,
                 augmented = self.dataConfig.augmented,
                 test_augmented = self.testConfig.augmented,
                 min_lr = self.trainConfig.min_lr,
                 max_lr = self.trainConfig.max_lr,
                )
        )

    def plateauConfig(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=25, eps=0)  # goal: maximize Dice score
        return
    def cyclicConfig(self):
        self.scheduler = optim.lr_scheduler.CyclicLR(optimizer, self.min_lr, self.max_lr, 500)
        return
    def eqweightStep(self):
        self.eqweight=kwargs['eqweight_decay']*self.eqweight
        return
        
    def basicImageLog(self, images, masks_pred, true_masks):
        self.experiment.log({
            'Images': wandb.Image(images[0].cpu()),
            'Masks' : {
                    'True' : wandb.Image(true_masks[0].float().cpu()),
                    'Predicted' : wandb.Image(masks_pred[0].argmax(dim=0).float().cpu())
                }
            })
        return
    def labelledImageLog(self, images, masks_pred, true_masks ):
        self.experiment.log({
            'images': wandb.Image(
                            images[0].cpu(),
                            caption =f"""Predicted = {masks_pred[0].argmax()}, 
                            True = {true_masks[0].argmax()}""" 
                      )
            })


    def basicTrainLog(self, epochTime, time0, epoch):
        if not self.trainConfig.debugging:
            self.experiment.log({
                'learning rate': self.optimizer.param_groups[0]['lr'],
                'step': self.global_step,
                'epoch': epoch,
                'Time into epoch': time.time()-epochTime,
                'Total time' : time.time()-time0,
                })

    def diceLoss(self, masks_pred, true_masks):
        return 1-dicescore(masks_pred,
                           true_masks,
                           num_classes = self.trainConfig.num_classes,
                           round = False,
                           average_classes = True) 

    def configEquivariance(self):
        if self.trainConfig.etransforms:
            return EquivarianceRegularizer(
                self.model,
                self.trainConfig.in_shape,
                self.trainConfig.etransforms,
                dist = 'cross_entropy_logits',
                n = self.trainConfig.n,
                num_funcs = self.trainConfig.num_funcs,
                bounds = self.trainConfig.bounds,
                )
        return lambda : 0

    def baseStep(self, term):
        self.optimizer.zero_grad(set_to_none=True)
        term.backward()
        self.optimizer.step()
        if self.scheduler != None:
            self.scheduler.step()
        return
        
    def UNetStep(self, term):
        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.scheduler !=None:
            self.scheduler.step()
        return

    def logDice(self, masks_pred, true_masks):
        self.experiment.log({'Training Dice' : self.calcDice(masks_pred, true_masks)})
        return
    def calcDice(self, masks_pred, true_masks):
        dice = dicescore(masks_pred,
                         true_masks,
                         num_classes = self.trainConfig.num_classes,
                         round = True,
                         average_classes = False
                        )
        return dice

    def logAccuracy(self, masks_pred, true_masks):
        self.experiment.log({'Accuracy' : self.calcAccuracy(masks_pred, true_masks)})
        return

    def calcAccuracy(self, masks_pred, true_masks):
        accuracy = float(torch.mean(
                            torch.eq(
                                torch.argmax(masks_pred, -1),
                                torch.argmax(true_masks, -1)
                            ).to(dtype=torch.float)
                        ))
        return accuracy

    def makeModel(self):
        return
