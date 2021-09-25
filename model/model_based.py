from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# class ModelBased(nn.Module):

#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, input_x):
#         pass


class SimpleMlp(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(SimpleMlp).__init__()
        self.pretrain = models.resnet18(pretrained = True)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.__freez_pretrain()

    # TODO: type annotation
    def forward(self, x) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    
    def __freez_pretrain(self) -> None:
        for param in self.pretrain.parameters():
            param.requires_grad = False
        

