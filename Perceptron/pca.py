from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt

data = load_wine()
y = data.target
X = data.data

# 1. 调用sklearn计算PCA
pca = PCA(n_components = 3)
reduced_X1 = pca.fit_transform(X)


# 2. 使用numpy计算PCA
cov = np.cov(np.array(X).T)
values, vectors = np.linalg.eig(cov)
vectors = vectors[:, :2]
reduced_X2 = np.dot(X, vectors)
print(reduced_X2)

red_x1, red_y1, red_x2, red_y2 = [], [], [], []
blue_x1, blue_y1, blue_x2, blue_y2 = [], [], [], []
green_x1, green_y1, green_x2, green_y2 = [], [], [], []

for i in range(len(reduced_X1)):
	if y[i] == 0:
		red_x1.append(reduced_X1[i][0])
		red_y1.append(reduced_X1[i][1])
		red_x2.append(reduced_X2[i][0])
		red_y2.append(reduced_X2[i][1])
	elif y[i] == 1:
		blue_x1.append(reduced_X1[i][0])
		blue_y1.append(reduced_X1[i][1])
		blue_x2.append(reduced_X2[i][0])
		blue_y2.append(reduced_X2[i][1])
	else:
		green_x1.append(reduced_X1[i][0])
		green_y1.append(reduced_X1[i][1])
		green_x2.append(reduced_X2[i][0])
		green_y2.append(reduced_X2[i][1])

plt.figure('PCA')

plt.subplot(121)
plt.title('calculate PCA by calling sklearn function')
plt.scatter(red_x1, red_y1, c='r', marker='x')
plt.scatter(blue_x1, blue_y1, c='b', marker='D')
plt.scatter(green_x1, green_y1, c='g', marker='.')

plt.subplot(122)
plt.title('calculate PCA by using numpy')
plt.scatter(red_x2, red_y2, c='r', marker='x')
plt.scatter(blue_x2, blue_y2, c='b', marker='D')
plt.scatter(green_x2, green_y2, c='g', marker='.')

plt.show()