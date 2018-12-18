"""
EECS 498 - Self Driving Cars
University of Michigan
Network for localization task
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LocalizationNet(nn.Module):

    def __init__(self):
        super(LocalizationNet, self).__init__()

        self.fc1 = nn.Linear(10, 7)
        self.fc2 = nn.Linear(7, 3)
        
        self.init_weights()

    def forward(self, h):
        h = F.relu(self.fc1(h))
        outputs = self.fc2(h)
        return outputs

    def init_weights(self):
        for fc in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(fc.weight)