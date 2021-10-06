import argparse
import collections
from model.model import Model
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils.types import LearningPhase
from dataloader import DatasetLoader, CategoriesSampler
# from torchviz import make_dot
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # setup train-loader
    train_dataset = DatasetLoader(LearningPhase.TRAIN, config.dataset_dir)
    train_sampler = CategoriesSampler(train_dataset.label, 
                                      config.num_batch, 
                                      config.way,
                                      config.shot + config.train_query)
    train_loader = DataLoader(dataset = train_dataset,
                              batch_sampler = train_sampler,
                              num_workers = 2,
                              pin_memory = True)
    # setup validation-loader
    val_dataset = DatasetLoader(LearningPhase.VAL, config.dataset_dir)
    val_sampler = CategoriesSampler(val_dataset.label,
                                    config.num_batch,
                                    config.way,
                                    config.shot + config.train_query)
    val_loader = DataLoader(dataset = val_dataset,
                            batch_sampler = val_sampler,
                            num_workers = 2,
                            pin_memory = True)
    
    # build model architecture, then print to console
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = Model(config.way, config.update_step).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.learner.parameters(), 'lr': config.learner_lr},
        {'params': model.meta_learner.parameters(), 'lr': config.meta_lr}
    ])

    # TODO: set lr scheduler
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model = model,                    
                      optimizer = optimizer,
                      device = device,
                      len_epoch = config.max_epoch,
                      data_loader = train_loader,
                      args = config,
                      valid_data_loader= val_loader)
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet']) # The network architecture
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['miniImageNet', 'tieredImageNet', 'FC100']) # Dataset
    parser.add_argument('--phase', type=str, default='meta_train', choices=['pre_train', 'meta_train', 'meta_eval']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='0') # GPU id
    parser.add_argument('--dataset_dir', type=str, default='./data/mini/') # Dataset folder

    # Parameters for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=100) # Epoch number for meta-train phase
    parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--shot', type=int, default=4) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, default=5) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task
    parser.add_argument('--meta_lr', type=float, default=0.01) # Learning rate for SS weights
    parser.add_argument('--learner_lr', type=float, default=0.005) # Learning rate for FC weights
    parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop
    parser.add_argument('--update_step', type=int, default=50) # The number of updates for the inner loop
    parser.add_argument('--step_size', type=int, default=10) # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.5) # Gamma for the meta-train learning rate decay
    parser.add_argument('--init_weights', type=str, default=None) # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=str, default=None) # The meta-trained weights for meta-eval phase
    parser.add_argument('--meta_label', type=str, default='exp1') # Additional label for meta-train

    # Parameters for pretain phase
    parser.add_argument('--pre_max_epoch', type=int, default=100) # TODO: 100 # Epoch number for pre-train phase
    parser.add_argument('--pre_batch_size', type=int, default=128) # TODO: 128 # Batch size for pre-train phase
    parser.add_argument('--pre_lr', type=float, default=0.1) # Learning rate for pre-train phase
    parser.add_argument('--pre_gamma', type=float, default=0.2) # Gamma for the pre-train learning rate decay
    parser.add_argument('--pre_step_size', type=int, default= 30) # TODO: 30 # The number of epochs to reduce the pre-train learning rate
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9) # Momentum for the optimizer during pre-train
    parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005) # Weight decay for the optimizer during pre-train


    args = parser.parse_args()
    # print(args)
    
    main(args)
