import torch.nn as nn
import segmentation_models_pytorch as smp


class MAtentionNet(nn.Module):

    def __init__(self, n_channels, n_class):
        super().__init__()

        self.conv  = smp.MAnet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=13,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
            )
      

    def forward(self, x):
        
        res = self.conv(x)
        
        return res
