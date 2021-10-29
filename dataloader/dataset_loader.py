# from torchvision import datasets, transforms
# from base import BaseDataLoader
from typing import Any, List, Tuple
from torch.functional import Tensor
from torch.utils.data import Dataset
from utils.types import LearningPhase 
import os.path as osp
import os
from PIL import Image
from torchvision import transforms
import numpy as np


class DatasetLoader(Dataset):

    def __init__(self, setname: LearningPhase, dataset_dir: str, train_augment: bool = False) -> None:
        PATH, label_list = self.__set_path(setname, dataset_dir)
        data, label = [], []
        folders_path = [osp.join(PATH, label_item) for label_item in label_list if os.path.isdir(osp.join(PATH, label_item))]
        
        for idx, folder_path in enumerate(folders_path):
            if idx > 2:
                break
            image_list_path = os.listdir(folder_path)
            for image_path in image_list_path:
                data.append(osp.join(folder_path, image_path))
                label.append(idx)
        
        self.data, self.label = data, label
        self.num_class = len(set(label))
        self.transform = self.__do_transform(train_augment)        


    def __set_path(self, setname: LearningPhase, dataset_dir: str) -> Tuple[str, List[str]]:
        if setname == LearningPhase.TRAIN:
            PATH = osp.join(dataset_dir, 'train')
        elif setname == LearningPhase.TEST:
            PATH = osp.join(dataset_dir, 'test')
        elif setname == LearningPhase.VAL:
            PATH = osp.join(dataset_dir, 'val')
        else:
            raise ValueError('Wrong setname')
        label_list = os.listdir(PATH)

        return PATH, label_list

    def __do_transform(self, train_augment: bool) -> transforms.Compose:
        image_size = 80
        if train_augment:
            transform = transforms.Compose([
                transforms.Resize(92),
                transforms.RandomResizedCrop(88),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ]) 
        return transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        path, label = self.data[idx], self.label[idx]
        print(f'{path}   -   {label}')
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

        