from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor
from dataReader import SparseMatrixDataset
import numpy as np

dataset = SparseMatrixDataset("../Dataset/data/data_sample.csv")

X = dataset[:][0].numpy()
y = dataset[:][1].numpy()


out = np.array([])
for a in y:
    out = np.append(out, a[0])

# X, y = datasets.load_digits(n_class=9, return_X_y=True)
# print(X, y, sep='\n')
data = X
clf = SGDRegressor()
feature_map_nystroem = Nystroem(gamma=0.25,
                                random_state=1,
                                n_components=7)
data_transformed = feature_map_nystroem.fit_transform(data)
clf.fit(data_transformed, out)
print([list(X[0])])
y_pred = clf.predict([list(X[0])])
print(y_pred)
print(y[0])

