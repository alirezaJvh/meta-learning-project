from dataloader import DatasetLoader, CategoriesSampler
from utils.types import LearningPhase

dataset = DatasetLoader(LearningPhase.TRAIN, './data/mini')

# print(dataset.label)
sampler = CategoriesSampler(dataset.label, 8, 5, 1)

for i in sampler:
    print(i)