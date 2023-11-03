from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor
from dataReader import SparseMatrixDataset
import numpy as np
import utils_func
import torch
from tqdm import tqdm

dataset = SparseMatrixDataset("../Dataset/data/all_format/all_format_Tesla-A100.csv")


X = dataset[:][0].numpy()
y = dataset[:][1].numpy()

print(f'X shape is {X.shape}')


out = np.array([])
for a in y:
    out = np.append(out, a[0])

# out = out.reshape(-1, 1)
print(f'out shape is {out.shape}')
print(out)

data = X
print(data)
clf = SGDRegressor(verbose=1, max_iter=1000, epsilon=0.01, early_stopping=True)
feature_map_nystroem = Nystroem(gamma=0.25, random_state=1, n_components=156)
data_transformed = feature_map_nystroem.fit_transform(data)
clf.fit(data_transformed, out)

# for i in range(10):
#     idx = utils_func.generate_random_int(0, len(dataset))
#     # print([list(X[idx])])
#     y_pred = clf.predict([list(X[idx])])
#     # print(y_pred)
#     # print(y[idx][0])
#     print(utils_func.MAPELoss(torch.tensor(y_pred), y[idx][0]))
#     print("---------------")
print(clf.score(data_transformed, out))


X, y = datasets.load_digits(n_class=9, return_X_y=True)
data = X/16
print(X, y)
clf = SGDRegressor()
feature_map_nystroem = Nystroem(gamma=.2,
                                random_state=1,
                                n_components=300)
data_transformed = feature_map_nystroem.fit_transform(data)
clf.fit(data_transformed, y)
print(clf.score(data_transformed, y))