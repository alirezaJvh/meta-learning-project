# from torchvision import datasets, transforms
# from base import BaseDataLoader
from typing import List, Tuple
from typing_extensions import ParamSpec
from torch.utils.data import Dataset
from utils.types import LearningPhase 
import os.path as osp
import os


# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)




class DatasetLoader(Dataset):

    def __init__(self, setname: LearningPhase, dataset_dir: str, train_augment: bool = False):
        
        PATH, label_list = self.__set_path(setname, dataset_dir)
        print('here')
        print(PATH)
        print('label')
        print(label_list)

    
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


        