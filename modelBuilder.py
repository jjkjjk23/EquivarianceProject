import torch
import torch.nn as nn
import numpy as np
import timm
from dinov2.layers import DINOHead, NestedTensorBlock as Block
from dinov2.models.vision_transformer import DinoVisionTransformer, vit_base, vit_large
from unet import UNet
from transformers import ASTFeatureExtractor, ASTModel, ASTConfig


class Transformer(nn.Module):
    def __init__(self, modelConfig, backbone_model, head):
        super().__init__()
        self.modelConfig = modelConfig
        self.head = head
        self.backbone_model = backbone_model

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone_model(x)
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
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
            head = 'DINO', #or 'custom'
            task = 'category',
            num_channels = 3,
            **kwargs
            ):
        self.device = device
        self.in_size = in_size
        self.num_classes = num_classes
        self.size = size
        self.out_shape = out_shape
        self.backBone = backBone
        self.head = head
        self.task = task
        self.num_channels = num_channels
        
        
class ModelBuilder:
    def __init__(self, modelConfig):
        self.modelConfig = modelConfig
        self.backbone_model = None
        return

    def build(self):
        if self.modelConfig.backBone == 'UNet':
            return self.buildBackbone()
        return Transformer(
                self.modelConfig,
                self.buildBackbone(),
                self.buildHead()) 

    def buildBackbone(self):
        if self.modelConfig.backBone == 'DINOold':
            return self.buildDINO()
        elif self.modelConfig.backBone == 'DINO':
            return self.buildDINO()
        elif self.modelConfig.backBone == 'ViT':
            return self.buildViT()
        elif self.modelConfig.backBone == 'UNet':
            return UNet(
                    self.modelConfig.num_channels,
                    self.modelConfig.num_classes
                    )
        elif self.modelConfig.backBone == 'AST':
            #myASTConfig = ASTConfig()
            model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            return model
        return

    def buildDINO(self):
        BACKBONE_SIZE = self.modelConfig.size # in ("small", "base", "large" or "giant")
        backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
        "small_reg": "vits14_reg4",
        "base_reg": "vitb14_reg",
        "large_reg": "vitl14_reg4",
        "giant_reg": "vitg14_reg4",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        backbone_model.eval()
        backbone_model.to(self.modelConfig.device)
        return backbone_model

    def buildDINOv2(self): 
        model = DinoVisionTransformer(
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            block_fn=Block,
            num_register_tokens=0,
            img_size = 526,
            block_chunks = 0,
        )
        model.load_state_dict(torch.load('dinov2_vitl14_pretrain.pth'))
        model = model.to(device = self.modelConfig.device)
        return model


    def buildViT(self):
        model = timm.create_model(
            "vit_base_patch16_224.augreg_in1k",
            pretrained=True,
            in_chans = 3).to(self.modelConfig.device).forward_features
        return lambda x : model(x)[0,0,:]

    def buildHead(self):
        if self.modelConfig.head == 'custom':
            return nn.Linear(
                    self.modelConfig.in_size,
                    np.prod(self.modelConfig.out_shape)
                    ) 
        elif self.modelConfig.head == 'DINO' and self.modelConfig.task == 'category':
            return DINOHead(
                    in_dim = self.modelConfig.in_size,
                    out_dim = 37,
                    hidden_dim = 512
                            ) 
        elif self.modelConfig.backBone == 'UNet':
            return lambda x: x
        else:
            return lambda x: x
