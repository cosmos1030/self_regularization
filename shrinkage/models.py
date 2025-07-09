import torch
import torch.nn as nn

# ------------------------------------------------------------
# 3. miniAlexNet
# ------------------------------------------------------------
class miniAlexNet(nn.Module):
    def __init__(self, num_classes, num_input_channel):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channel,96,5,3,2), 
            nn.ReLU(),
            nn.MaxPool2d(3,1,1),
            nn.Conv2d(96,256,5,3,2), 
            nn.ReLU(),
            nn.MaxPool2d(3,1,1))
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4,384), 
            nn.ReLU(),
            nn.Linear(384,192)    , 
            nn.ReLU(), 
            nn.Linear(192,num_classes))
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,1)
        return self.classifier(x)