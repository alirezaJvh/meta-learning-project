import collections
from typing import Tuple
import torch
from torch.functional import Tensor
from model.learner import Learner
from model.meta_learner import MetaLearner
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys


class BaseLearner(nn.Module):
    def __init__(self, way, z_dim):
        super().__init__()
        self.way = way
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc_w = nn.Parameter(torch.ones([self.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc_w)
        self.vars.append(self.fc_w)
        self.fc_b = nn.Parameter(torch.ones([self.way]))
        self.vars.append(self.fc_b)        

    def forward(self, input_x, the_vars = None):
        if the_vars is None:
            the_vars = self.vars
        fc_w = the_vars[0]
        fc_b = the_vars[1]
        net = F.linear(input_x, fc_w, fc_b)
        return net

    def parameters(self):
        return self.vars



class Pretrain_Maml(nn.Module):
    def __init__(self, way, update_step, learner_lr):
        super(Pretrain_Maml, self).__init__()
        self.way = way
        self.update_step = update_step
        # self.mode = mode
        self.learner_lr = learner_lr
        self.pretrain = models.resnet18(pretrained = True)
        self.base_learner = BaseLearner(way = self.way, z_dim = 1000)

        for param in self.pretrain.parameters():
            param.requires_grad = False

    def forward(self, data):
        train_data, train_label, test_data = data
        train_embedding = self.pretrain(train_data)
        test_embedding = self.pretrain(test_data)
        return self.meta_forward(train_embedding, train_label, test_embedding)

    def meta_forward(self, 
                    train_embedding, 
                    train_label, 
                    test_embedding):
        logits = self.base_learner(train_embedding)
        loss = F.cross_entropy(logits, train_label)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.learner_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(test_embedding, fast_weights)        

        for _ in range(1, self.update_step):
            logits = self.base_learner(train_embedding, fast_weights)
            loss = F.cross_entropy(logits, train_label)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.learner_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(test_embedding, fast_weights)  
        return logits_q
        