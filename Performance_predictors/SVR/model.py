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
        self.clf = SGDRegressor(verbose=1, 
                                max_iter=1000, 
                                epsilon=epsilon, 
                                early_stopping=True,
                                shuffle=True)
        self.feature_map_nystroem = Nystroem(kernel=kernel,
                                             gamma=gamma,
                                             random_state=1,
                                             n_components=7)
        self.feature_map_RBF = RBFSampler(gamma=gamma, n_components=7)
        self.linearSVR = LinearSVR(epsilon=epsilon, C=C, verbose=1, max_iter=99999999, intercept_scaling=1.0, loss="epsilon_insensitive", random_state=None, tol=0.0001)
        self.usualSVR = SVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon, verbose=True)
    
    def forward(self, x):
        return self.usualSVR.predict(x)

def train_SVR_Nystroem(model:SvrPredictor, dataset):
    X = dataset[:][0].numpy()
    Y = dataset[:][1].numpy()
    out = np.array([])
    for a in Y:
        out = np.append(out, a[0])

    print("shape of input data :", X.shape)
    data_transformed = model.feature_map_nystroem.fit_transform(X)

    model.clf.fit(data_transformed, out)
    print("score of model :", model.clf.score(data_transformed, out))
    return model

def train_LinearSVR(model:SvrPredictor, dataset):
    X = dataset[:][0].numpy()
    Y = dataset[:][1].numpy()
    out = np.array([])
    for a in Y:
        out = np.append(out, a[0])
    model.linearSVR.fit(X, out)
    print("score of model :", model.linearSVR.score(X, out))

def train_usualSVR(model:SvrPredictor, dataset):
    X = dataset[:][0].numpy()
    Y = dataset[:][1].numpy()
    out = np.array([])
    for a in Y:
        out = np.append(out, a[0])
    
    model.usualSVR.fit(X, out)
    print("score of model :", model.usualSVR.score(X, out))