from model.learner import Learner
from model.meta_learner import MetaLearner
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Model(nn.Module):

    def __init__(self,mode = 'meta-train',num_class = 64) -> None:
        super(Model, self).__init__()
        self.pretrain = models.resnet18(pretrained = True)
        self.meta_learner = MetaLearner(input_dim = 1000, output_dim = 64)
        self.learner = Learner(input_dim = 1000)
        self.__freeze_pretrain()

    def forward(self, data):
        # meta-train
        data_shot, label_shot, data_query = data
        embedding_query = self.pretrain(data_query)
        embdedding_shot = self.pretrain(data_shot)
        # task_parameter = self.learner(data_shot)
        print('task parameter')

    def __freeze_pretrain(self) -> None:
        for param in self.pretrain.parameters():
            param.requires_grad = False


