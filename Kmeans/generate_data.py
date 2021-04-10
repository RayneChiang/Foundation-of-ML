import numpy as np
from sklearn.datasets import make_spd_matrix
import pandas as pd
import matplotlib.pyplot as plt


def genGaussianSamples(N, m, C):
    A = np.linalg.cholesky(C)
    U = np.random.randn(N, 2)

    return U @ A.T + m


def gaussian_2D_value(x, m, C):
    c_inv = np.linalg.inv(C)
    c_det = np.linalg.det(C)
    num = np.exp(-0.5 * np.dot((x - m).T, np.dot(c_inv, (x - m))))
    den = 2 * np.pi * np.sqrt(c_det)
    return num / den


def gaussian_contour(x, y, m, c):
    n = len(x)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            xvec = np.array([X[i, j], Y[i, j]])
            #Z[i, j] = gaussian_2D_value(xvec, m, c) * 1000
            Z[i,j] =X[i, j]**2 + Y[i, j]**2
    return X, Y, Z


if __name__ == '__main__':
    # Define three means
    Means = np.array([[0, 3], [3, 0], [4, 4]])

    # Define three convariance matrices ensuring
    # they are positive definite
    # generate a symmetric matrix
    CovMatrices = np.zeros((3, 2, 2))
    for j in range(3):
        CovMatrices[j, :, :] = make_spd_matrix(2)

    # Priors
    w = np.random.rand(3)
    w = np.round(w / np.sum(w), decimals=3)

    # How many data in each component
    nData = np.floor(w * 1000).astype(int)

    X0 = genGaussianSamples(nData[0], Means[0, :], CovMatrices[0, :, :])
    X1 = genGaussianSamples(nData[1], Means[1, :], CovMatrices[1, :, :])
    X2 = genGaussianSamples(nData[2], Means[2, :], CovMatrices[2, :, :])
    Y0 = [0] * nData[0]
    Y1 = [1] * nData[1]
    Y2 = [2] * nData[2]

    # Append into an array for the data we need
    X = np.append(np.append(X0, X1, axis=0), X2, axis=0)
    Y = np.append(np.append(Y0, Y1, axis=0), Y2, axis=0)
    data = pd.DataFrame(data=X, columns=['X_axis', 'Y_axis'])
    label = pd.DataFrame(data=Y, columns=['label'])

    # data.to_csv('data.csv', columns=['X_axis', 'Y_axis'], index=None)
    # label.to_csv('label.csv', columns=['label'], index=None)
    X0_x, X0_y = data[:len(X0)]['X_axis'], data[0:len(X0)]['Y_axis']
    X1_x, X1_y = data[len(X0):len(X0) + len(X1)]['X_axis'], data[len(X0):len(X0) + len(X1)]['Y_axis']
    X2_x, X2_y = data[len(X0) + len(X1):]['X_axis'], data[len(X0) + len(X1):]['Y_axis']
    X_0, Y_0, Z_0 = gaussian_contour(X0_x, X0_y, Means[0, :], CovMatrices[0, :, :])
    X_1, Y_1, Z_1 = gaussian_contour(X1_x, X1_y, Means[1, :], CovMatrices[1, :, :])
    X_2, Y_2, Z_2 = gaussian_contour(X2_x, X2_y, Means[2, :], CovMatrices[2, :, :])

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(X0_x, X0_y, c='blue', s=1)
    ax.scatter(X1_x, X1_y, c='red', s=1)
    ax.scatter(X2_x, X2_y, c='purple', s=1)
    ax.contour(X_0, Y_0, Z_0, levels=5, cmap='Accent_r')
    ax.contour(X_1, Y_1, Z_1, levels=5, cmap='Accent')
    ax.contour(X_2, Y_2, Z_2, levels=5, cmap='Blues')

    ax.set_title('three classes of gaussian data')
    # plt.savefig('gaussian_3_data')
    plt.show()
