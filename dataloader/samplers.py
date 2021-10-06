from typing import Generator, List
import torch
import numpy as np
from torch.functional import Tensor


class CategoriesSampler():

    def __init__(self, label: List[int], num_batch: int, way: int, shot: int) -> None:
        self.num_batch = num_batch
        self.way = way
        self.shot = shot

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self) -> int:
        return self.num_batch

    def __iter__(self) -> Generator[Tensor, None, None]:
        for _ in range(self.num_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.way]
            for c in classes:
                class_data = self.m_ind[c]
                pos = torch.randperm(len(class_data))[:self.shot]
                batch.append(class_data[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch