from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class OptBased(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(OptBased).__init__()
        self.pretrain = models.resnet18(pretrained = True)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.__freez_pretrain()
    
    def forward(self, x, landa_paramter) -> Tensor:
        # TODO: landas parameter ?
        x = self.pretrain(x)
        x = F.relu(self.fc1(x))
        return x

    def __freez_pretrain(self) -> None:
        for param in self.pretrain.parameters():
            param.requires_grad = False



