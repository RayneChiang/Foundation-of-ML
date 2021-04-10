import numpy as np
import pandas as pd
import os
import random


def random_init_list(start, stop, k):
    mean_list = []
    for i in range(k):
        mean_list.append([random.random() * (stop - start), random.random() * (stop - start)])
    return mean_list


def compute_distortion_measure(K, X, Y_predict, process_Means):
    J = 0
    X = np.array(X)
    process_Means = np.array(process_Means)
    for n in range(K):
        index_list = Y_predict[Y_predict['label'] == n].index.tolist()
        for i in index_list:
            J += np.linalg.norm(X[i] - process_Means[n])
    return J


def update_means(K, X, Y_predict):
    updated_mean = []
    for n in range(K):
        index_list = Y_predict[Y_predict['label'] == n].index.tolist()
        X_n = X.iloc[index_list]
        X_axis = np.mean(X_n['X_axis'])
        Y_axis = np.mean(X_n['Y_axis'])
        updated_mean.append([X_axis, Y_axis])
    return updated_mean


def update_predict(X, process_mean):
    predict_list = []
    X = np.array(X)
    for i in X:
        means = []
        for mean in process_mean:
            means.append(np.linalg.norm(i - mean))
        predict_list.append(np.argmin(means))
    return predict_list


def k_means_implement(K, X_data):
    mean_init = np.array(random_init_list(X_data.min(), X_data.max(), K))
    predict_init = pd.DataFrame(data=np.random.randint(K, size=len(X_data)), columns=['label'])
    print(compute_distortion_measure(K, X_data, predict_init, mean_init))
    updated_predict = predict_init
    J = 0
    while True:
        updated_mean = update_means(K, X_data, updated_predict)
        updated_predict = pd.DataFrame(data=update_predict(X_data, updated_mean), columns=['label'])
        J_previous = J
        J = compute_distortion_measure(K, X_data, updated_predict, updated_mean)
        if J == J_previous:
            break

    return J, updated_mean, updated_predict


if __name__ == '__main__':
    np.random.seed(3)
    path = os.path.dirname(__file__)
    k = 3
    data = pd.read_csv(path + '/data.csv')
    target = pd.read_csv(path + '/label.csv')
    Means = np.array([[0, 3], [3, 0], [4, 4]])
    J_original = compute_distortion_measure(3, data, target, Means)
    Y_label = target
    means_list = [[Means, J_original]]

    for i in range(1, 7):
        print(i)
        J, result_mean, result_predict = k_means_implement(i, data)
        Y_label = np.append(Y_label, result_predict, axis=1)
        means_list.append([result_mean, J])
        print(J)

    Y_mean = pd.DataFrame(data=means_list, columns=['result_mean', 'J'])
    Y_mean.to_csv('iteration_mean.csv', index=None)
    Y_label = pd.DataFrame(data=Y_label)
    Y_label.to_csv('iteration_label.csv', index=None)
