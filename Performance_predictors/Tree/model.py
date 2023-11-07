from sklearn.tree import DecisionTreeRegressor
import torch
import numpy as np

class TreePredictor(torch.nn.Module):
    def __init__(self, max_deapth) -> None:
        super(TreePredictor, self).__init__()

        self.tree = DecisionTreeRegressor(max_depth=max_deapth)
    def forward(self, x):
        self.tree.predict(x)

def train_TreePredictor(model:TreePredictor, dataset):
    X = dataset[:][0].numpy()
    print(type(X))