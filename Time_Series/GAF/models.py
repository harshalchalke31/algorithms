import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CNN(nn.Module):
    def __init__(self,in_channels:int=1,num_classes:int=5):
        super(CNN,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=16,kernel_size=3,padding=1),
            nn.AvgPool2d(kernel_size=2), # 187 -> 93
            nn.ReLU(),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.AvgPool2d(kernel_size=2),  # 93 -> 46
            nn.ReLU(),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.AvgPool2d(kernel_size=2),   # 46 -> 23
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*23*23,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=num_classes)            
        )
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def build_resnet(in_channels:int=1,num_classes:int=5):
    model = resnet50(pretrained=True)
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)

    # initialize new weights from conv1 weights (first two channels of RGB)
    with torch.no_grad():
        if in_channels == 1:
            # Average the pretrained RGB weights to initialize the 1-channel conv
            model.conv1.weight[:, 0] = original_conv.weight.mean(dim=1)
        elif in_channels == 2:
            model.conv1.weight[:, :2] = original_conv.weight[:, :2]
        elif in_channels == 3:
            model.conv1.weight[:, :3] = original_conv.weight[:, :3]
        else:
            nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

    # replace final classifier layer
    model.fc = nn.Linear(in_features=model.fc.in_features,out_features=num_classes)

    return model
       
