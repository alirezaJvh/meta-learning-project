from typing import Tuple
import torch
from torch.functional import Tensor
from model.learner import Learner
from model.meta_learner import MetaLearner
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Model(nn.Module):

    def __init__(self,update_step: int, num_class: int, mode = 'meta-train') -> None:
        super(Model, self).__init__()
        self.mode = mode
        self.pretrain = models.resnet18(pretrained = True)
        self.meta_learner = MetaLearner(input_dim = 1000, output_dim = num_class)
        self.learner = Learner(input_dim = 1000)
        self.update_step = update_step
        self.__freeze_unfreeze_param(self.pretrain)

    def forward(self, data: Tuple[Tensor, Tensor, Tensor]):
        # meta-train
        train_data, train_label, test_data = data
        print('train_data')
        print(train_data.size())
        test_embedding = self.pretrain(test_data)
        train_embedding = self.pretrain(train_data)
        if(self.mode == 'meta-train'):
            return self.meta_train_forward(train_embedding, train_label, test_embedding)
        else:
            # meta-test phase
            pass
    
    def meta_train_forward(self, 
                           train_embedding: Tensor, 
                           train_label: Tensor, 
                           test_embedding: Tensor) -> Tensor:
        # train meta-train
        for _ in range(self.update_step):
            landa_param = self.learner(train_embedding)
            print('landa')
            print(landa_param.size())
            # parameter without bias
            meta_learner_param = next(self.meta_learner.parameters())
            # freeze meta_learner param
            self.__freeze_unfreeze_param(self.meta_learner)
            meta_learner_param += landa_param
            train_predict = self.meta_learner(train_embedding)
            loss = F.cross_entropy(train_predict, train_label)
            loss.backward()  
        # freeze learner param
        self.__freeze_unfreeze_param(self.learner)
        # unfreeze meta-learner param
        self.__freeze_unfreeze_param(self.meta_learner, required_grad = True)        
        test_predict = self.meta_learner(test_embedding)
        return test_predict
        
    def __freeze_unfreeze_param(self, model, required_grad = False) -> None:
        for param in model.parameters():
            param.requires_grad = required_grad

