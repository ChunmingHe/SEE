from lib.Modules import GCM3, GPM, REM11, BasicConv2d
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.Modules2 import *
'''
backbone: resnet50
'''

class Network_interFA_noSpade_noEdge_ODE_slot_channel4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128, imagenet_pretrained=True):
        super(Network_interFA_noSpade_noEdge_ODE_slot_channel4, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.gcm_interFA = GCM_interFA_ODE_slot_channel4(256, channels)
        self.GPM = GPM()
        self.rem_decoder = REM_decoder_noSpade_noEdge(channels)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        f1, f2, f3, f4 = self.gcm_interFA(x1, x2, x3, x4)
        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        prior_cam = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        _, f4, f3, f2, f1 = self.rem_decoder([f1, f2, f3, f4], prior_cam, image)
        return prior_cam, f4, f3, f2, f1

class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=96):
        super(Network, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM()
        self.REM11 = REM11(channels, channels)

        self.LL_down = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        )
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.one_conv_f4_ll = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        HH_up = self.dePixelShuffle(HH)  
        f1_HH = torch.cat([HH_up, f1], dim=1)  
        f1_HH = self.one_conv_f1_hh(f1_HH) 

        LL_down = self.LL_down(LL)  
        f4_LL = torch.cat([LL_down, f4], dim=1) 
        f4_LL = self.one_conv_f4_ll(f4_LL) 

        prior_cam = self.GPM(x4) 
        pred_0 = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM11([f1_HH, f2, f3, f4_LL], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


if __name__ == '__main__':
    image = torch.rand(2, 3, 384, 384).cuda()
    model = Network(96).cuda()
    pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = model(image)
    print(pred_0.shape)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(bound_f4.shape)
    print(bound_f3.shape)
    print(bound_f2.shape)
    print(bound_f1.shape)
