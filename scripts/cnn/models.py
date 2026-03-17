import torch.nn as nn
import torch
from torchvision import models


def get_resnet9(num_classes=10):
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.convs = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels)
            )
        def forward(self, x):
            return x + self.convs(x)

    def conv_block(in_c, out_c, pool=True):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    model = nn.Sequential(
        conv_block(1, 64, pool=False),           # 112 -> 112
        conv_block(64, 128, pool=True),          # 112 -> 56
        ResBlock(128),
        conv_block(128, 256, pool=True),         # 56 -> 28
        conv_block(256, 512, pool=True),         # 28 -> 14
        ResBlock(512),

        nn.AdaptiveMaxPool2d(1), 
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )

    # model_inference = nn.Sequential(model, nn.Softmax(dim=1))

    # return model, model_inference

    return model




def get_resnet18(num_classes=10, pretrained=False):
    # Load the model
    # model = models.resnet18(weights=None)
    model = models.resnet18(weights='DEFAULT')
    
    # Modify the first conv layer from rgb input to gray (originally 3, 64, kernel_size=7...) 
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # output modification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # model_inference = nn.Sequential(model, nn.Softmax(dim=1))
    # return model, model_inference
    return model

def get_resnet34(num_classes=10, pretrained=False):
    # Load the model
    # weights = 'DEFAULT' if pretrained else None
    # model = models.resnet34(weights=None)
    model = models.resnet18(weights='DEFAULT')
    
    # Modify the first conv layer from rgb input to gray (originally 3, 64, kernel_size=7...) 
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # output modification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # model_inference = nn.Sequential(model, nn.Softmax(dim=1))
    # return model, model_inference
    return model



