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

class Trainer():
    """
    Trainer class
    """
    def __init__(self, 
                 model: Model, 
                 optimizer: torch.optim,  
                 device: str,
                 args: dict,
                 data_loader: DataLoader, 
                 valid_data_loader: DataLoader, 
                 lr_scheduler = None) -> None:
        super().__init__()
        self.args = args
        self.model = model        
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.__run_log_path()        

    def train(self):
        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        writer = SummaryWriter(self.args.save_path)
        global_count = 0

        train_label = self.__set_label(self.args.shot)
        for epoch in range(1, self.args.max_epoch):
            self.model.train()
            test_label = self.__set_label(self.args.train_query)
            tqdm_gen = tqdm.tqdm(self.data_loader)
            self.__train_epoch(epoch, tqdm_gen, train_label, test_label, writer, global_count)


    def __train_epoch(self, 
                      epoch: int, 
                      tqdm_gen: tqdm, 
                      train_label: Tensor, 
                      test_label: Tensor,
                      writer: SummaryWriter,
                      global_count: int) -> None:
        for i, batch in enumerate(tqdm_gen, 1):
            global_count += 1
            if self.device == 'cuda':              
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            # split train and test data of task
            train_data, test_data = self.__split_task_data(data)
            test_predict = self.model((train_data, train_label, test_data))
            self.model.freeze_learner()
            loss = F.cross_entropy(test_predict, test_label)
            acc = self.count_acc(test_predict, test_label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.unfreeze_learner()

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


    def __set_label(self, repeat: int) -> Tensor:
        label = torch.arange(self.args.way).repeat(repeat)
        if self.device == 'cuda':
            label = label.type(torch.cuda.LongTensor)
        else:            
            label = label.type(torc__set_run_pathh.LongTensor)
        return label

    def __split_task_data(self, data) -> Tuple[Tensor, Tensor]:
        train_index = self.args.shot * self.args.way
        train_data, test_data = data[:train_index], data[train_index:]
        return train_data, test_data
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def __run_log_path(self):
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
            
        print(save_path2)
        self.args.save_path = f'{meta_base_dir}/{save_path1}_{save_path2}'
        if os.path.exists(self.args.save_path):
            pass
        else:
            os.mkdir(self.args.save_path)