from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

class MetaLearner(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x) -> Tensor:
        # TODO: landas parameter ?
        x = F.relu(self.fc1(x))
        return x


