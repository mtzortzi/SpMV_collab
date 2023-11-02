from sklearn.svm import SVR
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import Nystroem

class SvrPredictor(torch.nn.Module):
    def __init__(self, kernel, 
                 C,
                 epsilon,
                 gamma):
        super(SvrPredictor, self).__init__()
        self.clf = SGDRegressor(verbose=1)
        self.feature_map_nystroem = Nystroem(kernel=kernel,
                                             gamma=gamma,
                                             random_state=1,
                                             n_components=7)
        self.regressor = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    
    def forward(self, x):
        return self.clf.predict(x)

def train_SVR(model:SvrPredictor, dataset):
    X = dataset[:][0].numpy()
    Y = dataset[:][1].numpy()
    out = np.array([])
    for a in Y:
        out = np.append(out, a[0])

    data_transformed = model.feature_map_nystroem.fit_transform(X)

    model.clf.fit(data_transformed, out)
    print("score of model :", model.clf.score(data_transformed, out))
    return model