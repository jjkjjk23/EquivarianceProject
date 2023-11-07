import torch
import time
from etrainerfunctions import dicescore
from equivariance_regularizer_test import EquivarianceRegularizer
import wandb
#TODO: TestConfig

class TrainerConfig:
    def __init__(
            self,
            debugging = False,
            etransforms = None,
            equivariant = False,
            eqweight = None,
            n = 0,
            class_weights = None,
            save_checkpoint = False,
            min_lr = 0,
            max_lr = .01,
            wandb_project = None,
            eqweight_scheduler = False,
            task = 'category',
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            epochs = 1,
            amp = True,
            batchSize = 1,
            num_classes = 1,
            gradient_clipping = 1.0,
            dataset = 'HeLa',
            scheduler = None,
            optimizer = 'SGD',
            gradientScaling = False,
            learning_rate = .1,
            debugTest = True,
            in_shape = (3,224,224),
            loss = 'myLoss',
            endTest = False,
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
        self.in_shape = (3,224,224)
        self.loss = loss
        self.endTest = endTest
            
class TrainBuilder:
    def __init__(
            self,
            dataConfig,
            modelConfig,
            trainConfig,
            testConfig,
            train_loader,
            testLoader,
            model,
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

