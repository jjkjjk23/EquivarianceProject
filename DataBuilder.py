import torch
import torchvision
import numpy as np
import PIL
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from etrainerfunctions import RandomDataset, CombinedDataset
from transformers import ASTFeatureExtractor
from scipy.io import wavfile
import torch._dynamo
torch._dynamo.config.suppress_errors=True

"""
class MusicDataset(torch.utils.data.Dataset):
    def __init__(self,
                 trainDirectory,
                 labelDirectory,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available()
                            else 'cpu'
                 ),
                 numClasses=88
                 ):
        super().__init__()
        self.wav, self.fs = wavfile.read(trainFile)
        self.featureExtractor
        self.spectrograms = 
    

    def __getitem__(self,idx):
"""
@torch.compile
def oxfordInputTransform(image_not_tensor, dataConfig, device):
    totensor = torch.compile(torchvision.transforms.PILToTensor())
    resize = torchvision.transforms.Resize(dataConfig.in_shape[-2:], antialias=True)
    dinoNorm = torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                                std = (0.229, 0.224, 0.225)
                                                )
    ViTNorm = torchvision.transforms.Normalize(mean = (.5, .5, .5),
                                                std = (.5, .5, .5)
                                                )

    image = totensor(image_not_tensor)
    image = resize(image)
    image = image/255
    image = image.to(dtype = torch.float32, device = device)
    if dataConfig.backBone == 'DINO':
        image = dinoNorm(image)
    if dataConfig.backBone == 'ViT':
        image = ViTNorm(image)
    return image


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
        
    @torch.compile
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

    @torch.compile
    def __getitem__(self, idx):
        image, target = dataset[idx]
        return self.inputTransform(image), self.outputTransform(target)

class DataConfig:
    def __init__(
            self,
            dataset = 'Oxford',
            augmented = False,
            afunctions = None,
            task = 'category',
            backBone = 'DINO', #Equals 'DINO', 'ViT', or 'UNet' so far
            split = 'test', #'test' or 'trainval'
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            batchSize = 1,
            num_classes = 3,
            in_shape = (3,224,224),
            ):
        self.dataset = dataset
        self.augmented = augmented
        self.afunctions = afunctions
        self.task = task
        self.backBone = backBone
        self.split = split
        self.device = device
        self.batchSize = batchSize
        self.num_classes = num_classes
        self.in_shape = in_shape

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
            return oxfordInputTransform(image, self.dataConfig, self.dataConfig.device)
        if self.dataConfig.dataset == 'Music':
            return self.musicInput(image)
        return

    def musicInput(self, image):
        wav, fs = wavfile.read

    def oxfordInput(self, image):
        totensor = torchvision.transforms.PILToTensor()
        resize = torchvision.transforms.Resize(self.dataConfig.in_shape[-2:], antialias=True)
        dinoNorm = torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                                    std = (0.229, 0.224, 0.225)
                                                    )
        ViTNorm = torchvision.transforms.Normalize(mean = (.5, .5, .5),
                                                    std = (.5, .5, .5)
                                                    )

        image = totensor(image)
        image = resize(image)
        image = image/255
        image = image.to(dtype = torch.float32, device = self.dataConfig.device)
        if self.dataConfig.backBone == 'DINO':
            image = dinoNorm(image)
        if self.dataConfig.backBone == 'ViT':
            image = ViTNorm(image)
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
        totensor = torch.compile(torchvision.transforms.PILToTensor())
        resize = torchvision.transforms.Resize(self.dataConfig.in_shape[-2:], antialias=True)
        tensor = totensor(tensor)
        tensor = resize(tensor)
        tensor = tensor.to(dtype = torch.float32, device = self.dataConfig.device)

        tensor = tensor.squeeze(1)-1
        tensor = tensor.to(torch.int64)
        tensor = F.one_hot(tensor, self.dataConfig.num_classes)
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.to(dtype = torch.float32)
        tensor = tensor.squeeze(0)
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
        if self.dataConfig.augmented == 'randcombo':
            return RandomDataset(dataset0)

        return dataset0

    def configDataLoader(self, dataset):
        if self.split == 'trainval' and self.dataConfig.dataset=='Oxford':
            #            return self.valSplitDataLoader(dataset)
            return DataLoader(dataset,
                              shuffle = True,
                              batch_size = self.dataConfig.batchSize
                              )
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

    def valSplitDataLoader(self, dataset):
        n_val = 200
        n_train = len(dataset) - n_val
        train_set, _ = random_split(
                dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(0)
                )
        train_loader = DataLoader(
                train_set,
                shuffle=True,
                batch_size=self.dataConfig.batchSize
                )
        return train_loader

"""dataConfig = DataConfig()
dataBuilder = DataBuilder(dataConfig)
dataLoader = dataBuilder.makeDataLoader()
#print(dataConfig.dataset)
for batch in dataLoader:
    print(batch[0])
"""

