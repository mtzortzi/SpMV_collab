import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor, plot_tree

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80,1), axis=1)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Ploting the results
plt.figure()
plt.scatter(X, y, s=20, edgecolors="black", c="darkorange", label="data", zorder=1)
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2, zorder=-1)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2, zorder=-1)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.savefig("results.png")

plt.clf()

plot_tree(regr_1)
plt.savefig("tree1.png")
plt.clf()
plot_tree(regr_2)
plt.savefig("tree2.png")

