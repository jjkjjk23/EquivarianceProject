import torch
import torch.nn as nn
import numpy as np

class Transformer(nn.Module):
    def __init__(self, modelConfig, backbone_model, head):
        super().__init__()
        self.modelConfig = modelConfig
        self.head = head
        self.backbone_model = backbone_model

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone_model(x)
        x = self.head(x)
        x = x.reshape((x.shape[0],)+self.modelConfig.out_shape)
        return x


        
class ModelConfig:
    def __init__(
            self, 
            device = None, 
            in_size = 768,
            num_classes = 37, 
            size = "base",
            out_shape = (37,),
            backBone = 'DINO',
            ):
        self.device = device
        self.in_size = in_size
        self.num_classes = num_classes
        self.size = size
        self.out_shape = out_shape
        self.backBone = backBone
        
class ModelBuilder:
    def __init__(self, modelConfig):
        self.modelConfig = modelConfig
        self.backbone_model = None
        return

    def build(self):
        return Transformer(
                self.modelConfig,
                self.buildBackbone(),
                self.buildHead()) 

    def buildBackbone(self):
        if self.modelConfig.backBone == 'DINO':
            return self.buildDINO()
        return

    def buildDINO(self):
        BACKBONE_SIZE = self.modelConfig.size # in ("small", "base", "large" or "giant")
        backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        backbone_model.eval()
        backbone_model.to(self.modelConfig.device)
        return backbone_model

    def buildHead(self):
        return nn.Linear(
                self.modelConfig.in_size,
                np.prod(self.modelConfig.out_shape)
                ) 
        
    
