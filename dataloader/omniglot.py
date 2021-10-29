from typing import Any, List, Tuple
from torch.functional import Tensor
from torch.utils.data import Dataset
from utils.types import LearningPhase 
import os.path as osp
import os
from PIL import Image
from torchvision import transforms
import numpy as np



class Omniglot(Dataset):
    def __init__(self, root, transform = None, target_transform=None):
        self.root = root
        self.transform = transform
        self.all_items = self.find_classes(os.path.join(self.root))
        self.idx_classes = self.index_classes(self.all_items)
        # print(self.all_items)
        self.target_transform = target_transform


    def find_classes(self, root_dir):
        retour = []
        for (root, dirs, files) in os.walk(root_dir):
            # print(f'route: {root}, dirs: {dirs}, files: {files}')
            for f in files:
                if (f.endswith('png')):
                    r = root.split('/')
                    lr = len(r)
                    retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
                # print(retour)
        print(f'== Found {len(retour)} items')
        return retour


    def index_classes(self, items):
        idx = {}
        for i in items:
            if i[1] not in idx:
                idx[i[1]] = len(idx)
        print("== Found %d classes" % len(idx))
        return idx


    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # target is label
        return img, target

    def __do_transform(self):
        pass

    def __len__(self):
        return len(self.all_items)