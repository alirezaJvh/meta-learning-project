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
        self.optimizer = optimizer
        # self.save_path = self.__set_save_path()

    def train(self):
        # writer = SummaryWriter(self.save_path)
        train_label = self.__set_label(self.args.shot)
        # self.args.max_epoch
        for epoch in range(1, self.args.max_epoch):
            self.model.train()
            test_label = self.__set_label(self.args.train_query)
            tqdm_gen = tqdm.tqdm(self.data_loader)
            self.__train_epoch(epoch, tqdm_gen, train_label, test_label)


    def __train_epoch(self, 
                      epoch: int, 
                      tqdm_gen: tqdm, 
                      train_label: Tensor, 
                      test_label: Tensor) -> None:
        for i, batch in enumerate(tqdm_gen, 1):
            if self.device == 'cuda':              
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
                # split train and test data of task
            train_data, test_data = self.__split_task_data(data)
            train_mean, test_mean = self.__mean_data(train_data, test_data)
            # print('mean')
            # print(train_mean.size())
            # print(test_mean.size())
            # print('label')
            # print(train_label.size())
            # print(test_label.size())
            test_predict = self.model((train_mean, train_label, test_mean))
            self.model.freeze_learner()
            # print('label size')
            # print(test_predict.size())
            # print(test_label.size())
            loss = F.cross_entropy(test_predict, test_label)
            acc = self.count_acc(test_predict, test_label)
            tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.unfreeze_learner()
            # print('label size')
            # print(test_predict.size())
            # print(test_label.size())
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
        # label = torch.arange(self.args.way).repeat(repeat)
        label = torch.arange(self.args.way)
        if self.device == 'cuda':
            label = label.type(torch.cuda.LongTensor)
        else:            
            label = label.type(torch.LongTensor)
        return label

    def __split_task_data(self, data) -> Tuple[Tensor, Tensor]:
        train_index = self.args.shot * self.args.way
        train_data, test_data = data[:train_index], data[train_index:]
        return train_data, test_data

    def __mean_data(self, train_data: Tensor, test_data: Tensor) -> Tuple[Tensor, Tensor]:
        train  = torch.tensor([train_data[way::self.args.way].cpu().detach().numpy() for way in range(self.args.way)]).cuda()
        test  = torch.tensor([test_data[way::self.args.way].cpu().detach().numpy() for way in range(self.args.way)]).cuda()
        return torch.mean(train, 1), torch.mean(test, 1)

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
