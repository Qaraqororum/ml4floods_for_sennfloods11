import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

class MAtentionNet(nn.Module):

    def __init__(self, n_channels, n_class):
        super().__init__()

        self.conv  = smp.MAnet(        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",
            in_channels=n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class)

      

    def forward(self, x):
        
        res = self.conv(x)
        
        return res
class pspnet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()
        self.conv = smp.PSPNet(
            encoder_weights="imagenet",
            in_channels=n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class)

    def forward(self, x):
        
        res = self.conv(x)
        return res
class linknet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()
        self.conv = smp.Linknet(
            encoder_weights="imagenet",
            in_channels=n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class)

    def forward(self, x):
        
        res = self.conv(x)
        return res
        
class Unet_drop_extra(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()
        self.conv = smp.Unet(
            encoder_weights="imagenet",
            in_channels=n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class )

    def forward(self, x):
        
        res = self.conv(x)
        return res
