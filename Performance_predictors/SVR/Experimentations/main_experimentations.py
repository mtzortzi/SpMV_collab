import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# svm.SVR is a class that implements SVR. The hyperparameters are kernel function, C and ε.
# svm.LinSVR is similar to SVR class with parameter kernel=’linear’ but has a better performance for large datasets. The hyperparameters are C and ε.
# svm.NuSVR uses a parameter nu that controls the number of support vectors and complexity of model. Similar to SVR class, the hyperparameters are kernel function, C and ε.
from matplotlib import pyplot as plt

df = pd.read_csv("Student_Marks.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42)


# Train and test datasets are sorted for plotting purpose
train = train.sort_values('time_study')
test = test.sort_values('time_study')

X_train, X_test = train[['time_study', 'number_courses']], test[['time_study', 'number_courses']]
y_train, t_test = train['Marks'], test['Marks']

print(X_train, y_train)

# Feature Scaling
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)

# Fitting SVR model
svr_lin = SVR(kernel = 'linear')
svr_rbf = SVR(kernel = 'rbf')
svr_poly = SVR(kernel = 'poly')

svr_lin.fit(X_train_scaled, y_train)
svr_rbf.fit(X_train_scaled, y_train)
svr_poly.fit(X_train_scaled, y_train)

# Evaluating model performance

#### Model prediction for train dataset ####
train['linear_svr_pred'] = svr_lin.predict(X_train_scaled)
train['rbf_svr_pred'] = svr_rbf.predict(X_train_scaled)
train['poly_svr_pred'] = svr_poly.predict(X_train_scaled)

#### Visualization ####
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(train['time_study'], train['number_courses'], train['Marks'])
ax.scatter(train['time_study'], train['number_courses'], train['linear_svr_pred'], color = 'orange', label = 'linear SVR')
ax.scatter(train['time_study'], train['number_courses'], train['rbf_svr_pred'], color = 'green', label = 'rbf SVR')
ax.scatter(train['time_study'], train['number_courses'], train['poly_svr_pred'], color = 'blue', label = 'poly SVR')
ax.legend()
ax.set_xlabel('Study time')
ax.set_ylabel('Number courses')
ax.set_zlabel('Marks')
plt.savefig("temp.png")

