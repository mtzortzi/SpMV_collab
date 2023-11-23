from sklearn.svm import SVR
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

class SvrPredictor(torch.nn.Module):
    def __init__(self, kernel, 
                 C,
                 epsilon,
                 gamma):
        super(SvrPredictor, self).__init__()
        self.usualSVR = SVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon, verbose=True)
    
    def forward(self, x):
        return self.usualSVR.predict(x)

def train_usualSVR(model:SvrPredictor, dataset, out_feature):
    assert out_feature == 0 or out_feature == 1
    X = dataset[:][0].numpy()
    Y = dataset[:][1].numpy()
    out = np.array([])
    for a in Y:
        out = np.append(out, a[out_feature])
    
    model.usualSVR.fit(X, out)