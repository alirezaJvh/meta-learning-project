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

class Trainer():
    """
    Trainer class
    """
    def __init__(self, 
                 model: Model, 
                 optimizer: torch.optim,  
                 device: str,
                 args: dict,
                 len_epoch: int,
                 data_loader: DataLoader, 
                 valid_data_loader: DataLoader, 
                 lr_scheduler = None) -> None:
        super().__init__()

        self.model = model        
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.args = args
        # self.save_path = self.__set_save_path()

    def train(self):
        # writer = SummaryWriter(self.save_path)
        train_label, test_label = self.__set_label()
        tqdm_gen = tqdm.tqdm(self.data_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            if self.device == 'cuda':
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
        # split train and test data of task
        train_data, test_data = self.__split_task_data(data)
        predict = self.model((train_data, train_label, test_data))

        print(predict)


    def __set_label(self) -> Tuple[Tensor, Tensor]:
        test_label = torch.arange(self.args.way).repeat(self.args.val_query)
        train_label = torch.arange(self.args.way).repeat(self.args.shot)
        if self.device == 'cuda':
            test_label = test_label.type(torch.cuda.LongTensor)
            train_label = train_label.type(torch.cuda.LongTensor)
        else:
            test_label = test_label.type(torch.LongTensor)
            train_label = train_label.type(torch.LongTensor)
        return train_label, test_label

    def __split_task_data(self, data) -> Tuple[Tensor, Tensor]:
        train_index = self.args.shot * self.args.way
        train_data, test_data = data[:train_index], data[train_index:]
        return train_data, test_dataembedding_query

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

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
