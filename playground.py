from dataloader import DatasetLoader, Omniglot, CategoriesSampler
from utils.types import LearningPhase
from torch.utils.data import DataLoader
from model.maml import BaseLearner
import torch
import argparse


# dataset = DatasetLoader(LearningPhase.TRAIN, './data/mini')

# sampler = CategoriesSampler(dataset.label, 8, 5, 1)

# loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers= 2, pin_memory=True)

# x = torch.ones(5, requires_grad = True)
# y = x**2
# w = x.clone().detach()
# z = x**3

# r = (y+z).sum()
# r.backward()

# print(x.grad)

def main(config):
    omniglot = Omniglot(config.dataset_dir)
    print('omniglot')
    for i, data in enumerate(omniglot):
        if i < 2:
            print(data)
            print('*****')

    imageNet = DatasetLoader(LearningPhase.TRAIN, './data/mini/')

    # for i, data in enumerate(imageNet):
    #     if i < 1:
    #         print(data)
    #         print('*****')

    # val_sampler = CategoriesSampler(val_dataset.label,
    #                                 config.num_batch,
    #                                 config.way,
    #                                 config.shot + config.train_query)    
    # val_loader = DataLoader(dataset = val_dataset,
    #                     batch_sampler = val_sampler,
    #                     num_workers = 2,
    #                     pin_memory = True)


    # t = iter(val_loader)
    # data = t.next()
    # print(data[0].size())

    # test = torch.load('saved/log/meta/epoch-10')
    # model = BaseLearner(2, 1000)
    # print(test)







if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet']) # The network architecture
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['miniImageNet', 'tieredImageNet', 'FC100']) # Dataset
    parser.add_argument('--phase', type=str, default='meta_train', choices=['pre_train', 'meta_train', 'meta_eval']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='0') # GPU id    
    parser.add_argument('--dataset_dir', type=str, default='./data/omniglot/') # Dataset folder

    # Parameters for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=100) # Epoch number for meta-train phase
    parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--shot', type=int, default=5) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, default=2) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task
    parser.add_argument('--meta_lr', type=float, default=0.003) # Learning rate for SS weights
    parser.add_argument('--learner_lr', type=float, default=0.01) # Learning rate for FC weights

    args = parser.parse_args()
    print(f'way: {args.way}, shot: {args.shot}')
    main(args)
