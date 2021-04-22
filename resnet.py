import torch.nn as nn
import torchvision


def make_resnet(num_classes):
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(2048, num_classes, bias=True)
    return model
