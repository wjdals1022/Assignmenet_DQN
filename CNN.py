import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.LayerNorm2d1 = nn.LayerNorm([16, 20, 20])

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  
        self.LayerNorm2d2 = nn.LayerNorm([32, 9, 9])

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1) 
        self.LayerNorm2d3 = nn.LayerNorm([64, 7, 7])  

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1) 
        self.LayerNorm2d4 = nn.LayerNorm([128, 5, 5])  

        self.in_features = 128 * 5 * 5
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        # Conv layer 1
        x = self.conv1(x)
        x = self.LayerNorm2d1(x)
        x = self.activation(x)
        
        # Conv layer 2
        x = self.conv2(x)
        x = self.LayerNorm2d2(x)
        x = self.activation(x)

        # conv layer 3
        x = self.conv3(x)
        x = self.LayerNorm2d3(x)
        x = self.activation(x)

        # conv layer 4
        x = self.conv4(x)
        x = self.LayerNorm2d4(x)
        x = self.activation(x)

        # fully connected layers
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.activation(x) 
        x = self.fc2(x)  

        return x
