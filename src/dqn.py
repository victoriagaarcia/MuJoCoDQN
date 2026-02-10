import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# QNetwork (CNN+MLP)
class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions): 
        super(QNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
    
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)  # Assuming input shape is (4, 84, 84)
            n_flat = self.encoder(dummy).shape[1]
            
    
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)