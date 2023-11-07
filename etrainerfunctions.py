import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import time
import math
import functools

import wandb

import PIL
from torch.autograd import grad
import numpy as np
import random
def gradient(tensor, model):
    param = random.choice(list(model.parameters()))
    #if tensor.is_cuda:
    #    v = v.cuda()
    #print("Tensor shape, ", tensor.shape, "Parameter shape, ", x.shape, "Random vector shape, ", v.shape)
    #v always has shape (num_proj, shape)
    Jv = grad(tensor, param, grad_outputs=None, retain_graph=True, create_graph=True, is_grads_batched=False)[0]
    return torch.linalg.norm(Jv)

def _random_vector(shape):
    '''
    creates a random vector of dimension C with a norm of C^(1/2)
    (as needed for the projection formula to work)
    '''
    v = torch.randn(np.prod(shape))
    v = torch.nn.functional.normalize(v, dim=-1)
    return v.reshape(shape)

class HeLaDataset(torch.utils.data.Dataset):
    def __init__(self, train_file, label_file, n_images, transform=None, target_transform=None):
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
        if self.transform:
            return self.transform(torch.unsqueeze(torch.from_numpy(np.array(self.images)),0)), self.target_transform(torch.unsqueeze( torch.from_numpy(np.array(self.labels)),0))
        else:
            return torch.unsqueeze(torch.from_numpy(np.array(self.images)),0), torch.unsqueeze(torch.from_numpy(np.array(self.labels)),0)

        #return self
    def __len__(self):
        return self.n_images


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.numused = [0 for j in self.datasets]
        self.notexhausted = datasets

    def __getitem__(self, idx):
        while self.datasets:
            dataset = random.choice(self.notexhausted)
            i=self.datasets.index(dataset)
            self.numused[i]+=1
            if self.numused[i]==len(dataset)-1:
                self.notexhausted.remove(dataset)
            if not self.notexhausted:
                self.notexhausted = self.datasets
            currindex=(idx-sum([num for j , num in enumerate(self.numused) if j!=i]))%len(dataset)
            return dataset.__getitem__(currindex)

        raise StopIteration
    #def __iter__(self):
        #return self
    def __len__(self):
        return sum([len(j) for j in self.datasets])
    def reset(self):
        self.notexhausted=self.datasets
        self.numused=[0 for j in self.datasets]
        
class ScaleJitteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __getitem__(self, idx):
        image, true_mask = self.dataset.__getitem__(idx)
        a=random.uniform(.25,4)
        image=torchvision.transforms.Resize((math.floor(224*a),math.floor(224*a)))(image)
        image=torchvision.transforms.Resize((224,224))(image)
        true_mask=torchvision.transforms.Resize((math.floor(224*a),math.floor(224*a)))(true_mask)
        true_mask=torchvision.transforms.Resize((224,224))(true_mask)
        return image, true_mask
    def __len__(self):
        return len(self.dataset)
    
class RandomAngleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, fill=0):
        self.dataset=dataset
        self.fill = fill
    def __getitem__(self, idx):
        image, true_mask = self.dataset.__getitem__(idx)
       
        angle=random.uniform(-10,10)
        image=torchvision.transforms.functional.rotate(image, angle, fill=self.fill)
        true_mask=torchvision.transforms.functional.rotate(true_mask, angle,fill=self.fill)
        return image, true_mask
    def __len__(self):
        return len(self.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deformation = torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).to(device=device, dtype=torch.float32)
deformation=torch.unsqueeze(deformation, dim=0)
deformation= torchvision.transforms.Resize((512,512), interpolation=TF.InterpolationMode.BICUBIC)(deformation)
ydeformation =torch.full([1,512,512], 0, device=device, dtype=torch.float32)
deformation = torch.stack([deformation, ydeformation], dim=3)


        
def shift(x, shiftnum=1, axis=-1):
    x=torch.transpose(x, axis, -1)
    if shiftnum == 0:
        padded = x
    elif shiftnum > 0:
      #paddings = (0, shift, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[1]=shiftnum
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., shiftnum:], paddings)
    elif shiftnum < 0:
        #paddings = (-shift, 0, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[0]=-shiftnum
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., :shiftnum], paddings)
    else:
        raise ValueError
    return torch.transpose(padded, axis,-1)
        
def shifty(shiftnum, axis):
    return lambda x : shift(x, shiftnum, axis)
def deform(tensor):
    return TF.elastic_transform(tensor, deformation, TF.InterpolationMode.NEAREST, 0.0)
    
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __getitem__(self, idx):
        image, true_mask = self.dataset.__getitem__(idx)
        #transform_or_not = random.randint(0,1)
        #if transform_or_not == 0:
            #return image, true_mask
#        stacked = torch.stack((image, true_mask), dim=0)
        angle=random.uniform(-10,10)
        shiftnum = random.randint(-6,6)
        axis = random.randint(-2,-1)
        
        functions = [shifty(shiftnum, axis), rotate(angle), torchvision.transforms.ElasticTransform(interpolation=TF.InterpolationMode.NEAREST)]
            
        randcombo = functools.reduce(compose, functions)
#        stacked=randcombo(stacked)
        image = randcombo(image)
        
        #image=stacked[0]
        #true_mask=stacked[1]
        return image, true_mask
    def __len__(self):
        return len(self.dataset)
    
    
#split must equal 'test' or 'trainval'
def config_data(aug_transforms=None, augmented=False, split='trainval', batch_size=1, **kwargs):
    totensor=torchvision.transforms.PILToTensor()
    resize=torchvision.transforms.Resize((224,224))
    resizedtensor=torchvision.transforms.Compose([resize,totensor])
    dinov2norm = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    dinov2transform = lambda image: dinov2norm(resize(totensor(image)).to(dtype=torch.float))
    if kwargs['Oxford']:
        dataset0 = torchvision.datasets.OxfordIIITPet(root="C:\\Users\\jjkjj\\Equivariant\\", split=split, target_types=kwargs['task'],
                                        download=True, transform=dinov2transform, target_transform=None if kwargs['task'] =='category' else resizedtensor)
    if kwargs['HeLa']:
        dataset0= HeLaDataset(f"C:\\Users\\jjkjj\\Equivariant\\ISBI-2012-challenge\\{split.removesuffix('val')}-volume.tif",f"C:\\Users\\jjkjj\\Equivariant\\ISBI-2012-challenge\\{split.removesuffix('val')}-labels.tif", 30)
    
    if augmented in [True, 'True no identity']:
        if kwargs['Oxford']:       
            transformed_datasets=[torchvision.datasets.OxfordIIITPet(root="C:\\Users\\jjkjj\\Equivariant\\", split=split, target_types=kwargs['task'], download=True, transform=None if kwargs['task'] =='category' else torchvision.transforms.Compose([resizedtensor,f[1]]), target_transform=torchvision.transforms.Compose([resizedtensor,f[0]])) for f in aug_transforms]
        if kwargs['HeLa']:  
            transformed_datasets=[HeLaDataset(f"C:\\Users\\jjkjj\\Equivariant\\ISBI-2012-challenge\\{split.removesuffix('val')}-volume.tif",f"C:\\Users\\jjkjj\\Equivariant\\ISBI-2012-challenge\\{split.removesuffix('val')}-labels.tif", 30, transform=f[1], target_transform=f[0]) for f in aug_transforms]
        if augmented == True:
            all_datasets=[dataset0]+transformed_datasets
            dataset=CombinedDataset(all_datasets)
        else:
            dataset = transformed_datasets[0] #should really be combined
  
    
        
    elif augmented==False:
        dataset=dataset0
    elif augmented=='random':
        dataset = ScaleJitteredDataset(dataset0)
    elif augmented=='rangle':
        dataset = RandomAngleDataset(dataset0, fill=0 if kwargs['HeLa'] else 1)
    elif augmented=='randcombo':
        dataset=RandomDataset(dataset0)
    
    # 2. Split into train / validation partitions
    if split=='trainval' and kwargs['Oxford']:
        n_val = 200
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=batch_size)
        return train_loader#, val_loader
   
    return DataLoader(dataset, shuffle=True if split=='trainval' else False, batch_size=batch_size)

               
def randomscale(tensor):
        a=random.uniform(.25,4)
        tensor=torchvision.transforms.Resize((math.floor(224*a),math.floor(224*a)))(tensor)
        return torchvision.transforms.Resize((224,224))(tensor)
    
def pad24(inputs):
        return torch.nn.functional.pad(inputs, (0,24,0,24), mode='constant', value=1)

def sampler(shape, bounds=None, n=100,cuda=False):
    shape=(n,)+shape
    if bounds==None:
        output= torch.from_numpy(np.random.default_rng().random(size=shape)).to(torch.float32)
    elif type(bounds)==list:
        bounds=np.array(bounds)
        bounds=np.broadcast_to(bounds,shape+(2,))
        output = torch.from_numpy(np.random.default_rng().random(size=shape)*(bounds[...,1]-bounds[...,0])+bounds[...,0]).to(torch.float32)
    if cuda==True:
        return output.cuda()
    else:
        return output

def integrate(integrand, shape, bounds, n=100, measure=1.0, error=False, cuda=False):
    values=integrand(sampler(shape, bounds, n, cuda=cuda))
    if error==True:
        return (measure*torch.mean(values, dim=0), measure*torch.std(values, dim=0)/math.sqrt(n))
    if error==False:
        return measure*torch.mean(values, dim=0)
    
def Linffn(integrand, shape, bounds, n=100, cuda=False):
    sampled_vals=sampler(shape, bounds, n, cuda=cuda)
    values=integrand(sampler(shape, bounds, n, cuda=cuda))
    return torch.max(values, dim=0).values

def compose(function2,function1):
    return lambda *args : function2(function1(*args))

def l2integrand(f):
    return lambda inputs : torch.sum((f[0](inputs)-f[1](inputs))**2)

def CEintegrand(f):
    return lambda inputs : torch.max(torch.abs(torch.log(f[0](inputs))-torch.log(f[1](inputs))))

def linfintegrand(f):
    return lambda inputs : torch.max(torch.abs(f[0](inputs)-f[1](inputs)))

def l1integrand(f):
    return lambda inputs : torch.mean(torch.abs(f[0](inputs)-f[1](inputs)))

def rotate(angle, fill=0):
    return lambda inputs : torchvision.transforms.functional.rotate(inputs, angle, fill=1)
    
def args2list(*args):
    if len(args)==1:
        return args
    else:
        return [arg for arg in args]
    
def expand0(tensor):
    return torch.unsqueeze(tensor, axis=0)

def e(inputs):
    return inputs

def zero(inputs):
    return 0*inputs

def shift(x, shift=1, axis=-1):
    x=torch.transpose(x, axis, -1)
    if shift == 0:
        padded = x
    elif shift > 0:
        #paddings = (0, shift, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[1]=shift
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., shift:], paddings)
    elif shift < 0:
        #paddings = (-shift, 0, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[0]=-shift
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., :shift], paddings)
    else:
        raise ValueError
    return torch.transpose(padded, axis,-1)
            
def model2func(model):
    return lambda inputs : model.forward(inputs)

def equivariance_error(model, eintegrand, functions, shape, bounds=None, n=50, cuda=False, Linf=False):
    equivariance_measures={'linf' : linfintegrand, 'l2' : l2integrand, 'l1' : l1integrand}
    if type(eintegrand)==str:
        eintegrand=equivariance_measures[eintegrand]
    if bounds != None:
        if type(bounds)==list:
            bounds=np.array(bounds)
        measure = np.prod((bounds[...,1]-bounds[...,0]))
    if bounds == None:
        measure=1
    realintegrand=lambda f: eintegrand([compose(f[0],model),compose(model,f[1])])
    if Linf==False:
        return [torch.nn.functional.relu(integrate(realintegrand(f), shape, bounds, n=n, measure=measure,cuda=cuda)-f[2]) for f in functions]
    else:
        return [torch.nn.functional.relu(Linffn(realintegrand(f), shape, bounds, n=n,cuda=cuda)-f[2]) for f in functions]

#Calculate dice score for one-hot encoded targets and predictions
def dicescore(pred,target, ignore_index=None, average_classes=None, class_dim=1, num_classes=None):
    assert pred.shape == target.shape
    
    pred = torch.transpose(pred, class_dim, -1)
    target = torch.transpose(target, class_dim, -1)
    if num_classes !=1:
        pred = torch.argmax(pred, dim=-1)
        pred = F.one_hot(pred,  num_classes=num_classes)
    else:
        pred=torch.round(torch.sigmoid(pred))
    split_pred = torch.split(pred,1,dim=-1)
    split_target = torch.split(target, 1, dim=-1)
    tp = [(x*split_target[j]).sum() for j,x in enumerate(split_pred) if j!=ignore_index]
    fp = [(x*(1-split_target[j])).sum() for j,x in enumerate(split_pred) if j!=ignore_index]
    fn = [((1-x)*(split_target[j])).sum() for j,x in enumerate(split_pred) if j!=ignore_index]
    scores=[2*tp[j]/(2*tp[j]+fp[j]+fn[j]) if tp[j]+fp[j]+fn[j] != 0 else 1 for j in range(len(tp))]
    if len(scores)==0:
        return 0
    if average_classes == None:
        return scores
    elif average_classes == True:
        return sum(scores)/len(scores)
    elif average_classes == 'weighted':
        return sum([scores[j]*(split_target[j].sum()) for j in range(len(scores))])/target.sum()

#If you're using mean instead of max you might not want infinity to be 0
def pointwise_equivariance_error(model, tensor, f, vector=False):
    tensor = torch.cat((tensor, 1+torch.exp(-model(tensor))), dim=0)
    tensor = f[0](tensor)
    m=tensor.shape[0]
    #precompose = 1+torch.exp(-model(tensor[0:m//2]))
    #postcompose = fval[m//2:m]
    if vector:
        return torch.nan_to_num(torch.log(1+torch.exp(-model(tensor[0:m//2])))-torch.log(fval[m//2:m]), posinf=0,neginf=0)
    if model.Linf == False:
        return torch.mean(torch.abs(torch.nan_to_num(torch.log(1+torch.exp(-model(tensor[0:m//2])))-torch.log(tensor[m//2:m]), posinf=0,neginf=0)))
    else: 
        return torch.max(torch.abs(torch.nan_to_num(torch.log(1+torch.exp(-model(tensor[0:m//2])))-torch.log(tensor[m//2:m]), posinf=0,neginf=0)))   
    
        
        
