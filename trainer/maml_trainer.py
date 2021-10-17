import os
import os.path as osp
from typing import Tuple
import numpy as np
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
import tqdm
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.model import Model
from utils import inf_loop, MetricTracker
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import sys



class MamlTrainer():
    def __init__(self, 
                 model, 
                 optimizer, 
                 device, 
                 args, 
                 data_loader, 
                 valid_data_loader):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader 
        self.vald_data_loader = valid_data_loader
        self.device = device

    def train(self):
        # TODO: log

        # writer = SummaryWriter(self.args.save_path)
        save_path = self.__set_run_path()
        self.writer = SummaryWriter(comment = save_path)
        train_label = self.__set_label(self.args.shot)
        # print(train_label)
        # sys.exit()
        global_count = 0
        for epoch in range(1, self.args.max_epoch):
            self.model.train()
            test_label = self.__set_label(self.args.train_query)
            tqdm_gen = tqdm.tqdm(self.data_loader)
            self.__train_epoch(epoch, tqdm_gen, train_label, test_label, global_count)
        self.writer.close()

    def __train_epoch():
        pass

    def __set_label(self, repeat):
        label = torch.arange(self.args.way).repeat(repeat)
        return label.to(self.device)

    def __train_epoch(self, epoch, tqdm_gen, train_label, test_label, global_count):
        for i, batch in enumerate(tqdm_gen, 1):
            global_count += 1
            if self.device == 'cuda':              
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            train_data, test_data = self.__split_task_data(data)
            test_predict = self.model((train_data, train_label, test_data))
            loss = F.cross_entropy(test_predict, test_label)
            acc = self.count_acc(test_predict, test_label)
            self.writer.add_scalar('data/loss', float(loss), global_count)
            self.writer.add_scalar('data/acc', float(acc), global_count)
            tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                torch.save(self.model.base_learner.parameters(), f'saved/log/meta/epoch-{epoch}')
            # print('finish first epoch')
            # print(self.model.base_learner.parameters())
            # sys.exit()
        
    def __split_task_data(self, data) -> Tuple[Tensor, Tensor]:
        train_index = self.args.shot * self.args.way
        train_data, test_data = data[:train_index], data[train_index:]
        return train_data, test_data

    def count_acc(self, logits, label):
        """The function to calculate the .
        Args:
        logits: input logits.
        label: ground truth labels.
        Return:
        The output accuracy.
        """
        pred = F.softmax(logits, dim=1).argmax(dim=1)
        if torch.cuda.is_available():
            return (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return (pred == label).type(torch.FloatTensor).mean().item()
    
    def __set_run_path(self):
        log_base_dir = './runs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([self.args.dataset, self.args.model_type])
        obj = {
            'learner_lr': self.args.learner_lr, 
            'meta_lr': self.args.meta_lr,
            'batch': self.args.num_batch,
            'update_step': self.args.update_step, 
            'epoch': self.args.max_epoch
        }
        save_path2 = ''
        for item in obj:
            save_path2 += f'_{str(item)}:{obj[item]}'
            
        save_path = f'{meta_base_dir}/{save_path1}_{save_path2}'
        print(save_path)
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        return save_path

