from sklearn.svm import SVR
import torch
import numpy as np
from tqdm import tqdm

class SvrPredictor(torch.nn.Module):
    def __init__(self, kernel, 
                 C,
                 epsilon,
                 gamma):
        super(SvrPredictor, self).__init__()
        self.regressor = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    
    def forward(self, x):
        return self.regressor.predict(x)

def train_SVR(model:SvrPredictor, dataset):
    X = dataset[:][0].numpy()
    Y = dataset[:][1].numpy()
    out = np.array([])
    for a in Y:
        out = np.append(out, a[0])
    model.regressor.fit(X, out)
    return model
    


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