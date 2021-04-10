# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge, lasso_path
from matplotlib import pyplot as plt
import numpy as np


def error(target, predict):
    error = 0
    for i in range(len(target)):
        error += target[i] - predict[i]
    return error

# load data, inspect and do exploratory plots
diabetes = datasets.load_diabetes()

X = diabetes.data
t = diabetes.target




NumData, NumFeatures = X.shape
print(NumData, NumFeatures)
print(t.shape)

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
# ax[0].hist(t, bins=40)
# ax[1].scatter(X[:, 6], X[:, 7], c='m', s=3)
# ax[1].grid(True)
# plt.tight_layout
# plt.savefig("DiabetesTargetAndTwoInputs.jpg")

# Linear regression using sklearn
lin = LinearRegression()
lin.fit(X, t)
th1 = lin.predict(X)

# Pseudo-increase solution to linear regression
w = np.linalg.inv(X.T @ X) @ X.T @ t
th2 = X @ w


# Pseudo-increase solution to linear regression with intercept
O = np.ones((len(X), 1))
X2 = np.append(X, O, axis=1)
w2 = np.linalg.inv(X2.T @ X2) @X2.T @t
th3 = X2 @ w2

# Plot linear predictions to check if they look the same
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax[0].set_xlabel("target")
ax[0].set_ylabel("predict")
ax[0].scatter(t, th1, c='c', s=3)
ax[1].set_xlabel("target")
ax[1].set_ylabel("predict")
ax[1].scatter(t, th2, c='g', s=3)
plt.savefig("Linear_regression.jpg")

# Advanced in Linear regression
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax[0].set_xlabel("target")
ax[0].set_ylabel("predict")
ax[0].scatter(t, th1, c='c', s=3)
ax[1].set_xlabel("target")
ax[1].set_ylabel("predict")
ax[1].scatter(t, th3, c='g', s=3)
plt.savefig("Linear_regression_advance.jpg")




# Tikhanov (quadratic) Regularizer
gamma = 0.2
wR = np.linalg.inv(X2.T @ X2 + gamma*np.identity(NumFeatures+1)) @ X2.T @ t

l1 = Lasso(alpha=0.2)
l1.fit(X, t)
th_lasso = l1.predict(X)
print(' L1 Reg:{:.3f}'.format(error(t, th_lasso)))
l2 = Ridge(alpha=0.2)
l2.fit(X, t)
th_ridge = l2.predict(X)
print(' L2 Reg:{:.3f}'.format(error(t, th_ridge)))

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 16))
ax[0].bar(np.arange(len(wR)), wR)  # Tikhanov (quadratic) Regularizer
ax[1].bar(np.arange(len(l2.coef_)), l2.coef_) # Ridge
ax[0].set_ylim(-900, 900)
ax[1].set_ylim(-900, 900)
ax[0].set_title("Tikhanov (quadratic) Regularizer")
ax[1].set_title("Ridge regularizer")
plt.savefig("compare L2 regularizer.jpg")



fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
ax[0].bar(np.arange(len(w)), w) # Pseudo-increase solution to linear regression
ax[1].bar(np.arange(len(l1.coef_)), l1.coef_)# Lasso
ax[2].bar(np.arange(len(l2.coef_)), l2.coef_)# Ridge
ax[0].set_ylim(-900, 900)
ax[1].set_ylim(-900, 900)
ax[2].set_ylim(-900, 900)
ax[0].set_title("Pseudo-increase solution")
ax[1].set_title("Lasso regularizer")
ax[2].set_title("Ridge regularizer")
plt.savefig("LeastSquaresAndRegularizedWeight.jpg")


N = 100
y = np.empty(0)
X = np.empty([0,6])
for i in range(N):
    Z1= np.random.randn()
    Z2= np.random.randn()
    y = np.append(y, 3*Z1 - 1.5*Z2 + 2*np.random.randn())
    Xarr = np.array([Z1,Z1,Z1,Z2,Z2,Z2])+ np.random.randn(6)/5
    X = np.vstack ((X, Xarr.tolist()))


alphas_lasso, coefs_lasso, _ = lasso_path(X, y, fit_intercept=False)
# Plot each coefficient
#
fig, ax = plt.subplots(figsize = (8,4))
for i in range(6):
    ax.plot(alphas_lasso, coefs_lasso[i,:])
ax.grid(True)
ax.set_xlabel("Regularization")
ax.set_ylabel("Regression Coefficients")

plt.savefig("regularizer_path.jpg")
plt.show()


