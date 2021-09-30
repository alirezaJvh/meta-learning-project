import torch
from model.learner import Learner
from model.meta_learner import MetaLearner
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Model(nn.Module):

    def __init__(self,update_step: int, mode = 'meta-train', num_class: int = 64) -> None:
        super(Model, self).__init__()
        self.mode = mode
        self.pretrain = models.resnet18(pretrained = True)
        self.meta_learner = MetaLearner(input_dim = 1000, output_dim = 64)
        self.learner = Learner(input_dim = 1000)
        self.update_step = update_step
        self.__freeze_unfreeze_param(self.pretrain)

    def forward(self, data):
        # meta-train
        data_shot, label_shot, data_query = data
        embedding_query = self.pretrain(data_query)
        embdedding_shot = self.pretrain(data_shot)
        if(self.mode == 'meta-train'):
            return self.meta_train_forward(embdedding_shot, label_shot, embedding_query)
        else:
            # meta-test phase
            pass
    
    def meta_train_forward(self, embedding_shot, label_shot, embedding_query):
        # train meta-train
        for _ in range(self.update_step):
            landa_param = self.learner(embedding_shot)
            # parameter without bias
            meta_learner_param = next(self.meta_learner.parameters()) 
            # freeze meta_learner param
            self.__freeze_unfreeze_param(self.meta_learner)
            meta_learner_param += landa_param
            train_predict = self.meta_learner(embedding_shot)
            loss = F.cross_entropy(train_predict, label_shot)
            loss.backward()  
        # freeze learner param
        self.__freeze_unfreeze_param(self.learner)
        # unfreeze meta-learner param
        self.__freeze_unfreeze_param(self.meta_learner, required_grad = True)        
        test_predict = self.meta_learner(embedding_query)
        return test_predict
        
    def __freeze_unfreeze_param(self, model, required_grad = False) -> None:
        for param in model.parameters():
            param.requires_grad = required_grad
        


