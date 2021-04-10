from sklearn.cluster import KMeans
from kmeans import update_means, compute_distortion_measure
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    path = os.path.dirname(__file__)
    data = pd.read_csv(path + '/data.csv')
    target = pd.read_csv(path + '/label.csv')
    Means = np.array([[0, 3], [3, 0], [4, 4]])

    Y_label = target

    J_original = compute_distortion_measure(3, data, target, Means)
    means_list = [[Means, J_original]]
    for i in range(1, 10):
        est = KMeans(n_clusters=i)
        est.fit(data)
        labels = est.labels_
        predict = pd.DataFrame(data=labels, columns=['label'])
        means = update_means(i, data, predict)
        J = compute_distortion_measure(i, data, predict, means)
        print(J)
        Y_label = np.append(Y_label, predict, axis=1)
        means_list.append([means, J])

    Y_mean = pd.DataFrame(data=means_list, columns=['result_mean', 'J'])
    Y_mean.to_csv('sklearn_mean.csv', index=None)
    Y_label = pd.DataFrame(data=Y_label)
    Y_label.to_csv('sklearn_label.csv', index=None)