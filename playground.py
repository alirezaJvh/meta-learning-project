from dataloader import DatasetLoader, CategoriesSampler
from utils.types import LearningPhase
from torch.utils.data import DataLoader

dataset = DatasetLoader(LearningPhase.TRAIN, './data/mini')

sampler = CategoriesSampler(dataset.label, 8, 5, 1)

loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers= 2, pin_memory=True)


for data in loader:
    print(data[0].size())
    break