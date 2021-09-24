from dataloader import DatasetLoader
from utils.types import LearningPhase

dataset = DatasetLoader(LearningPhase.TRAIN, './data/mini')

for i in dataset:
    print(i)