from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Learner(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super(Learner, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1000)


    # TODO: type annotation
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = x.view(1000, 64)
        return x
    
    # def __freez_pretrain(self) -> None:
    #     for param in self.pretrain.parameters():
    #         param.requires_grad = False
        

