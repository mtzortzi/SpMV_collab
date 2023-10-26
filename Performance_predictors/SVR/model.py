import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
import torch
from torch.utils.data import DataLoader


class SvrPredictor(torch.nn.Module):
    def __init__(self, kernel, 
                 C,
                 epsilon,
                 gamma):
        self.regressor = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    
    def forward(self, x):
        return self.regressor.predict(x)

def train_SVR(model:SvrPredictor, dataset):
    # TODO: retrieve dataset
    X = []
    Y = []

    sc_X = StandardScaler()
    sc_Y = StandardScaler()
    X = sc_X.fit_transform(X)
    Y = sc_Y.fit_transform(Y)
    model.regressor.fit(X, Y)

# dataset = pd.read_csv("./Position_Salaries.csv")
# X = dataset.iloc[:,1:2].values.astype(float)
# Y = dataset.iloc[:,2:3].values.astype(float)

# sc_X = StandardScaler()
# sc_Y = StandardScaler()
# X = sc_X.fit_transform(X)
# Y = sc_Y.fit_transform(Y)

# regressor = SVR(kernel='rbf', C=10, epsilon=0.01, gamma=0.25)
# regressor.fit(X, Y)
# array = np.array([])
# array = np.append(array, 11)
# y_pred = regressor.predict(array.reshape(1, -1))
# print(sc_Y.inverse_transform(np.array(y_pred).reshape(1, -1)))
# #y_pred = sc_Y.inverse_transform((regressor.predict(sc_X.transform(array.reshape(1, -1)))))
# Y_pred = regressor.predict(X)

# plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color = 'magenta')
# plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y_pred.reshape(-1, 1)), color = 'green')
# plt.title('Truth or Bluff (Support Vector Regression Model)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()