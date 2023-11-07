from sklearn.tree import DecisionTreeRegressor
import torch
import numpy as np

class TreePredictor(torch.nn.Module):
    def __init__(self, max_depth:int):
        super(TreePredictor, self).__init__()

        self.tree = DecisionTreeRegressor(max_depth=max_depth)
    def forward(self, x):
        return self.tree.predict(x)

def train_TreePredictor(model:TreePredictor, dataset):
    X : np.ndarray = dataset[:][0].numpy()
    Y : np.ndarray = dataset[:][1].numpy()
    y = np.array([])
    for a in Y:
        y = np.append(y, a[0])
    model.tree.fit(X, y)