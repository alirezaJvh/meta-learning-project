import collections
from typing import Tuple
import torch
from torch.functional import Tensor
from model.learner import Learner
from model.meta_learner import MetaLearner
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .maml import BaseLearner
import sys

class Model(nn.Module):

    def __init__(self, 
                 way: int,  
                 update_step: int,
                 mode = 'meta-train') -> None:
        super(Model, self).__init__()
        self.mode = mode
        self.pretrain = models.resnet34(pretrained = True)
        self.meta_learner = MetaLearner(input_dim = 1000, output_dim = way)
        self.learner = Learner(input_dim = 1000)
        self.update_step = update_step
        self.way = way
        self.learner_param, self.meta_learner_param = None, None
        # freeze pretrain
        for param in self.pretrain.parameters():
            param.requires_grad = False

    def forward(self, data: Tuple[Tensor, Tensor, Tensor]):
        # meta-train
        train_data, train_label, test_data = data
        train_embedding = self.pretrain(train_data)
        test_embedding = self.pretrain(test_data)

        if(self.mode == 'meta-train'):
            return self.meta_train_forward(train_embedding, train_label, test_embedding)
        else:
            #TODO: meta-test phase
            pass
    
    def meta_train_forward(self,
                           train_embedding: Tensor, 
                           train_label: Tensor, 
                           test_embedding: Tensor) -> Tensor:
        self.learner_param = self.learner.state_dict()
        self.meta_learner_param = self.meta_learner.state_dict()
        # train meta-train
        for _ in range(self.update_step):
            train_predict = self.__predict_model(train_embedding)
            loss = F.cross_entropy(train_predict, train_label)
            loss.backward() 
        test_predict = self.__predict_model(test_embedding)
        self.meta_learner.load_state_dict(self.meta_learner_param)       
        return test_predict

    def freeze_learner(self) -> None:
        new_learner_param = self.learner_param.copy()
        for key in new_learner_param.keys():
            new_learner_param[key] = new_learner_param[key].detach()
        self.learner.load_state_dict(new_learner_param)

    def unfreeze_learner(self) -> None:
        self.learner.load_state_dict(self.learner_param)

    def __set_meta_new_param(self, theta: Tensor, landa: Tensor) -> None:
        new_param = self.meta_learner_param.copy()
        new_param['fc1.weight'] = theta + landa
        self.meta_learner.load_state_dict(new_param)

    def __predict_model(self, embedding: Tensor) -> Tensor:
        theta_param = self.meta_learner_param['fc1.weight'].clone().detach()
        mean_embedding = self.__mean_data(embedding)
        landa_param = self.learner(mean_embedding)
        self.__set_meta_new_param(theta_param, landa_param)
        predict = self.meta_learner(embedding)
        return predict

    def __mean_data(self, data: Tensor) -> Tensor:
        data  = torch.tensor([data[way::self.way].cpu().detach().numpy() for way in range(self.way)]).cuda()
        return torch.mean(data, 1)




class FixedModel(nn.Module):
    def __init__(self, way, update_step, lr, mode = 'meta-train'):
        super(FixedModel, self).__init__()
        self.pretrain = models.resnet34(pretrained = True)
        self.way = way
        self.meta_learner = SimpleNN(way, 1000)
        self.learner = Learner(input_dim = 1000)
        self.update_step = update_step
        self.lr = lr
        self.mode = mode

        file = 'MiniImageNet_ResNet_2way_5shot__learner_lr:0.01_meta_lr:0.003_batch:100_update_step:100_epoch:100'
        self.meta_parameter = torch.load(f'./saved/logs/maml/{file}/max_acc')

        # for param in self.meta_parameter:
        #     param.requires_grad = False

        for param in self.pretrain.parameters():
            param.requires_grad = False

    def forward(self, data):
        train_data, train_label, test_data = data
        train_embedding = self.pretrain(train_data)
        test_embedding = self.pretrain(test_data)
        if(self.mode == 'meta-train'):
            return self.meta_train_forward(train_embedding, train_label, test_embedding)
        else:
            #TODO: meta-test phase
            pass

    def meta_train_forward(self, 
                           train_embedding, 
                           train_label, 
                           test_embedding):
        # for _ in range(1, 2):
        #     mean_embedding = self.__mean_data(train_embedding)
        #     landa_param = self.learner(mean_embedding)

        #     new_meta_param = nn.ParameterList()
        #     theta = nn.Parameter(self.meta_parameter[0].clone().detach() + landa_param)
        #     new_meta_param.append(theta)
        #     bias = nn.Parameter(self.meta_parameter[1].clone().detach())
        #     new_meta_param.append(bias)

        #     logits = self.meta_learner(train_embedding, new_meta_param)

        #     loss = F.cross_entropy(logits, train_label)
        #     grad = torch.autograd.grad(loss, new_meta_param)
        #     fast_weights = list(map(lambda p: p[1] - self.lr * p[0], zip(grad, new_meta_param)))

        mean_embedding = self.__mean_data(train_embedding)
        landa_param = self.learner(mean_embedding)
        new_meta_param = nn.ParameterList()
        theta = nn.Parameter(self.meta_parameter[0].clone().detach() + landa_param)
        new_meta_param.append(theta)
        bias = nn.Parameter(self.meta_parameter[1].clone().detach())
        new_meta_param.append(bias)
        self.meta_learner.set_param(new_meta_param)

        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr = self.lr)

        for _ in range(1, 2):
            pred = self.meta_learner(train_embedding)
            loss = F.cross_entropy(pred, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_pred = self.meta_learner(test_embedding)
        return test_pred
            



    def __mean_data(self, data: Tensor) -> Tensor:
        data = torch.tensor([data[way::self.way].numpy() for way in range(self.way)])
        # data  = torch.tensor([data[way::self.way].cpu().detach().numpy() for way in range(self.way)]).cuda()
        return torch.mean(data, 1)



class SimpleNN(nn.Module):
    def __init__(self, out_dim, in_dim):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    
    def set_param(self, params):
        self.linear.weight = params[0]
        self.linear.bias = params[1]

    def forward(self, input_x):

        # for param in self.linear.parameters():
        #     print(param.size())
        predic = self.linear(input_x)
        return predic
