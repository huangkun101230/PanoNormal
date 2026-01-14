import numpy as np
import os
import torch.nn as nn
import torchvision

from .normal_decoder import NormalDecoder
from .panoformer import Panoformer as encoder
from .VGG_encoder import VGG16_Feat
import matplotlib.pyplot as plt
from torchvision import transforms

        

class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        orig_vgg = torchvision.models.vgg16(pretrained = True)
        features = orig_vgg.features
        self.vgg16_feature_extractor = VGG16_Feat(features)
        # Freeze all layers except the bottleneck
        # for name, param in self.vgg16_feature_extractor.named_parameters():
        #     if "bottle" not in name:
        #         param.requires_grad = False

        self.encoder = encoder()
        self.normal_decoder = NormalDecoder()

    def forward(self, rgb_inputs):
        self.vgg_feat = self.vgg16_feature_extractor(rgb_inputs)
        # print("self.vgg_feat: ",self.vgg_feat.shape)
        # feat = self.vgg_feat.permute(0,2,1).reshape(1,32,256,512)
        # show_feature_map(feat)
        # exit()
        # print("self.vgg_feat: ",self.vgg_feat.shape)

        self.encodes = self.encoder(self.vgg_feat)

        self.pred_results = self.normal_decoder(self.encodes)

        outputs = {}
        outputs["pred_normal"] = self.pred_results[0]
        outputs["pred_multiscale_normal"] = self.pred_results

        return outputs