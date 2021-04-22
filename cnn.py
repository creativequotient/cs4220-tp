import torch
import torch.nn as nn
import torch.nn.functional as F

""" This is the 3 convolutional layer + 3 dense layer CNN as described
in the report """
class CNN(nn.Module):
    def __init__ (self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(3,5,5,1)
        self.conv2 = nn.Conv2d(5,20,3,1)
        self.conv3 = nn.Conv2d(20,20,3,1)

        self.fc1 = nn.Linear(20*30*30,120)
        self.fc2 = nn.Linear(120,80)
        self.fc3 = nn.Linear(80,num_classes)

    def forward(self,x):
        bs = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2,2)

        x = x.view(bs,-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim = 1)
        return(x)
