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

from utils.util import Averager



class FixedTrainer():
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
        self.val_loader = valid_data_loader
        self.device = device

    def train(self):
        log_folder = 'fixed_maml'
        self.__set_run_path(log_folder)
        self.__set_trlog(init = True)
        # set summary writer for tensorboard
        self.writer = SummaryWriter(comment = f'.{self.save_path}')

        train_label = self.__set_label(self.args.shot)
        global_count = 0

        
        for epoch in range(1, self.args.max_epoch):
            # train phase
            self.model.train()
            query_label = self.__set_label(self.args.train_query)
            tqdm_gen = tqdm.tqdm(self.data_loader)
            # train averager
            self.train_loss_avg = Averager()
            self.train_acc_avg = Averager()
            # train each epoch
            self.__train_epoch(epoch, tqdm_gen, train_label, query_label, global_count)
            # update train averager
            self.train_loss_avg = self.train_loss_avg.item()
            self.train_acc_avg = self.train_acc_avg.item() 
            # val averager
            self.val_loss_avg = Averager()
            self.val_acc_avg = Averager()
            # val phase
            self.model.eval()
            # test label
            test_label = self.__set_label(self.args.val_query)
            self.__val_epoch(train_label, test_label)
            # avgtrain_loss_avtrain_loss_avtrain_loss_av
            self.val_loss_avg = self.val_loss_avg.item()
            self.val_acc_avg = self.val_acc_avg.item()
            # tensorboar
            self.writer.add_scalar('data/val_loss', float(self.val_loss_avg), epoch)
            self.writer.add_scalar('data/val_acc', float(self.val_acc_avg), epoch)
            # print val
            print(f'Epoch {epoch}, Val, Loss={self.val_loss_avg:.4f} Acc={self.val_acc_avg:.4f}')

            if self.val_acc_avg > self.trlog['max_acc']:
                self.trlog['max_acc'] = self.val_acc_avg
                self.trlog['max_acc_epoch'] = epoch
                self.__save_model('max_acc')

            if epoch % 10 == 0:
                self.__save_model(epoch)

            self.__set_trlog(init = False)

        self.writer.close()


    def __set_label(self, repeat):
        label = torch.arange(self.args.way).repeat(repeat)
        return label.to(self.device)

    def __train_epoch(self, 
                      epoch, 
                      tqdm_gen, 
                      train_label, 
                      test_label, 
                      global_count):
        for i, batch in enumerate(tqdm_gen, 1):
            global_count += 1
            if self.device == 'cuda':              
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            # split data
            train_data, test_data = self.__split_task_data(data)
            test_predict = self.model((train_data, train_label, test_data))
            loss = F.cross_entropy(test_predict, test_label)
            acc = self.count_acc(test_predict, test_label)
            # train_average
            self.train_loss_avg.add(loss.item())
            self.train_acc_avg.add(acc)
            # tensorboard
            self.writer.add_scalar('data/loss', float(loss), global_count)
            self.writer.add_scalar('data/acc', float(acc), global_count)
            # progress bar
            tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()


    def __val_epoch(self,   
                    train_label,
                    test_label):
        for i, batch in enumerate(self.val_loader, 1):
            if self.device == 'cuda':
                data, _ = [_.cuda for _ in batch]
            else:
                data = batch[0]
            train_data, test_data = self.__split_task_data(data)
            logits = self.model((train_data, train_label, test_data))
            loss = F.cross_entropy(logits, test_label)
            acc = self.count_acc(logits, test_label)

            self.val_loss_avg.add(loss.item())
            self.val_acc_avg.add(acc)
            
        
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
    

    def __set_run_path(self, log_folder):
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, log_folder)
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
        # save_path = f'{save_path2}'
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        # return save_path
        self.save_path = f'{log_folder}/{save_path1}_{save_path2}'
        print(self.save_path)

    
    def __save_model(self, epoch):
        path = f'saved/logs/{self.save_path}'
        if not osp.exists(path):
            os.mkdir(path)
        torch.save(self.model.base_learner.parameters(), f'{path}/epoch-{epoch}')


    def __set_trlog(self, init = False):
        if init:
            self.trlog = {
                'args': vars(self.args),
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'max_acc': 0.0,
                'max_acc_epoch': 0
            }
        else:
            self.trlog['train_loss'].append(self.train_loss_avg)
            self.trlog['train_acc'].append(self.train_acc_avg)
            self.trlog['val_loss'].append(self.val_loss_avg)
            self.trlog['val_acc'].append(self.val_acc_avg)
