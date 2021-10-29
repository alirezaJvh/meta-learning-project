import argparse
import collections
from model.maml import Pretrain_Maml
from model.model import FixedModel
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from trainer import MamlTrainer
from trainer import FixedTrainer
from utils import prepare_device
from utils.types import LearningPhase
from dataloader import DatasetLoader, CategoriesSampler
import os

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
        # setup train-loader
    test_dataset = DatasetLoader(LearningPhase.TEST, config.dataset_dir)
    test_sampler = CategoriesSampler(test_dataset.label, 
                                      config.num_batch, 
                                      config.way,
                                      config.shot + config.val_query)
    test_loader = DataLoader(dataset = test_dataset,
                             batch_sampler = test_sampler,
                             num_workers = 2,
                             pin_memory = True)

    # build model architecture, then print to console
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu) 
    print('Using gpu:', config.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    model = Pretrain_Maml(config.way, config.update_step, config.learner_lr).to(device)

    # model = FixedModel(config.way, config.update_step, config.learner_lr).to(device)    

    optimizer = torch.optim.Adam(model.base_learner.parameters(), lr = config.meta_lr)  

    # optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.pretrain.parameters())}, \
    #     {'params': model.base_learner.parameters(), 'lr': 0.001}], lr= 0.0001)    

        
    trainer = MamlTrainer(model = model,                    
                          optimizer = optimizer,
                          device = device,
                          data_loader = train_loader,
                          args = config,
                          valid_data_loader= val_loader)

    # trainer = FixedTrainer(model = model,                    
    #                      optimizer = optimizer,
    #                      device = device,
    #                      data_loader = train_loader,
    #                      args = config,
    #                      valid_data_loader= val_loader)

    
    trainer.train()


if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet']) # The network architecture
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['miniImageNet', 'tieredImageNet', 'FC100']) # Dataset
    parser.add_argument('--phase', type=str, default='meta_train', choices=['pre_train', 'meta_train', 'meta_eval']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='1') # GPU id    
    parser.add_argument('--dataset_dir', type=str, default='./data/mini/') # Dataset folder

    # Parameters for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=200) # Epoch number for meta-train phase
    parser.add_argument('--num_batch', type=int, default=32) # The number for different tasks used for meta-train
    parser.add_argument('--shot', type=int, default=5) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, default=5) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task
    parser.add_argument('--meta_lr', type=float, default=0.005) # Learning rate for SS weights
    parser.add_argument('--learner_lr', type=float, default=0.1) # Learning rate for FC weights
    # parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop
    parser.add_argument('--update_step', type=int, default=10) # The number of updates for the inner loop
    parser.add_argument('--step_size', type=int, default=10) # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.5) # Gamma for the meta-train learning rate decay
    parser.add_argument('--init_weights', type=str, default=None) # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=str, default=None) # The meta-trained weights for meta-eval phase
    parser.add_argument('--meta_label', type=str, default='exp1') # Additional label for meta-train


    args = parser.parse_args()
    print(f'way: {args.way}, shot: {args.shot}')
    main(args)
