import torch
import torchvision
import numpy as np
import PIL
import torch.nn.functional as F
from torch.utils.data import DataLoader


class HeLaDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 train_file, 
                 label_file, 
                 n_images, 
                 transform=None, 
                 target_transform=None,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 num_classes = 1,
                 ):
        super().__init__()
        self.images=PIL.Image.open(train_file)
        self.images.load()
        self.labels=PIL.Image.open(label_file)
        self.labels.load()
        #self.tensor=torchvision.transforms.PILToTensor(PIL.Image.open(file))
        self.n_images=n_images
        self.transform=transform
        self.target_transform=target_transform
        
    def __getitem__(self, idx):
        self.images.seek(idx)
        self.labels.seek(idx)
        
        image = np.array(self.images)
        label = np.array(self.labels)
        
        image = self.processImage(image)
        label = self.processImage(label)
        
        if self.transform:
            return self.transform(image), self.target_transform(label)
        return image, label

    def processImage(self, image):
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(dtype = torch.float32, device = device)

    def __len__(self):
        return self.n_images

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, inputTranform, outputTransform):
        self.dataset = dataset
        self.inputTransform = inputTransform
        self.outputTransform = outputTranform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = dataset[idx]
        return self.inputTransform(image), self.outputTransform(target)

class DataConfig:
    def __init__(self,
             dataset = 'Oxford',
             augmented = False,
             afunctions = None,
             task = 'category',
             backBone = 'DINO', #Equals 'DINO' or 'UNet' so far
             split = 'test', #'test' or 'trainval'
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
             batchSize = 1,
             ):
        self.dataset = dataset
        self.augmented = augmented
        self.afunctions = afunctions
        self.task = task
        self.backBone = backBone
        self.split = split
        self.device = device
        self.batchSize = batchSize

class DataBuilder:
    def __init__(self, dataConfig):
        self.dataConfig=dataConfig
        if dataConfig.dataset == 'HeLa':
            self.split = self.dataConfig.split.removesuffix('val')
        else:
            self.split = self.dataConfig.split
    def makeDataLoader(self, test=False):
        dataset0 = self.configBaseDataset()
        dataset = self.augmentDatasets(dataset0)
        return self.configDataLoader(dataset)

    def inputTransform(self, image):
        if self.dataConfig.dataset == 'HeLa':
            return self.HeLaInput(image)
        if self.dataConfig.dataset == 'Oxford':
                return self.oxfordInput(image)
        return

    def oxfordInput(self, image):
        totensor = torchvision.transforms.PILToTensor()
        resize = torchvision.transforms.Resize((224,224))
        dinoNorm = torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                                    std = (0.229, 0.224, 0.225)
                                                    )
        image = totensor(image)
        image = resize(image)
        image = image.to(dtype = torch.float32, device = self.dataConfig.device)
        if self.dataConfig.backBone == 'DINO':
            image = dinoNorm(image)
        return image
    #Implement HeLa
    def outputTransform(self, image):
        if self.dataConfig.dataset == 'Oxford':
            if self.dataConfig.task == 'segmentation':
                return self.oxfordSegOutput(image)
            else:
                return self.oxfordCatOutput(image) 
        return image

    def oxfordSegOutput(self, tensor):
        tensor = tensor.squeeze(1)-1
        tensor = tensor.to(torch.int64)
        tensor = F.one_hot(tensor, self.dataConfig.num_classes)
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.to(dtype = torch.float32)
        return tensor

    def oxfordCatOutput(self,image):
        image = torch.tensor(image, dtype = int)
        image = F.one_hot(image, 37)
        image = image.to(dtype = torch.float32, device = self.dataConfig.device)
        return image

    def configBaseDataset(self):
        if self.dataConfig.dataset == 'Oxford':
            dataset0 = torchvision.datasets.OxfordIIITPet(
                    root="../",
                    split=self.split,
                    target_types = self.dataConfig.task,
                    download = True,
                    transform = self.inputTransform,
                    target_transform = self.outputTransform,
                    )
        elif self.dataConfig.dataset == 'HeLa':
            dataset0 = HeLaDataset(f"/../ISBI-2012-challenge/{self.split}-volume.tif",
                                   f"/../ISBI-2012-challenge/{self.split}-labels.tif",
                                   30,
                                   transform = self.inputTransform,
                                   target_transform = self.outputTransform,
                                   device = self.dataConfig.device,
                                   num_classes = self.dataConfig.num_classes,
                                  )
        return dataset0
    #Do the actual augmented ones
    def augmentDatasets(self, dataset0):
        if self.dataConfig.augmented==False:
            return dataset0
        return dataset0

    def configDataLoader(self, dataset):
        if self.split == 'trainval' and self.dataConfig.dataset=='Oxford':
            return self.valSplitDataLoader(dataset)
        if self.split == 'trainval' or self.split == 'train':
            return DataLoader(dataset,
                              shuffle = True,
                              batch_size = self.dataConfig.batchSize
                              )
        else:
            return DataLoader(dataset,
                              shuffle = False,
                              batch_size = self.dataConfig.batchSize
                              )

"""dataConfig = DataConfig()
dataBuilder = DataBuilder(dataConfig)
dataLoader = dataBuilder.makeDataLoader()
#print(dataConfig.dataset)
for batch in dataLoader:
    print(batch[0])
"""

