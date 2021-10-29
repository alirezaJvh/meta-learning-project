from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch

class Learner(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super(Learner, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1000)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # TDOO: change activation
        x = F.relu(self.fc4(x))
        return x
 

