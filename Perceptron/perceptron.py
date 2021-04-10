import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import urllib.request
import pandas as pd


def PercentCorrect(Inpusts, targets, weights, bias):
    '''
    calculate the percentage of correctly classified examples
    :param Inpusts:
    :param targets:
    :param weights:
    :return:
    '''
    N = len(targets)
    nCorrect = 0

    for n in range(N):
        oneInput = Inpusts.iloc[n, :].values
        # targets = pd.array(targets)
        if targets[n] * (np.dot(oneInput, weights) + bias) > 0:
            nCorrect += 1
    return 100 * nCorrect / N


def perceptron_learning(X_train, y_train, X_test, y_test, N_train, w):
    bias = 0
    y_train = y_train['class'].values
    y_test = y_test['class'].values
    print('Initial Percentage Correct :%6.2f' % (PercentCorrect(X_train, y_train, w, bias)))

    MaxIter = 10000

    alpha = 0.002

    P_train = np.zeros(MaxIter)
    P_test = np.zeros(MaxIter)
    count = 0
    for iter in range(MaxIter):
        r = np.floor(np.random.rand() * N_train).astype(int)
        x = X_train.iloc[r, :].values
        if y_train[r] * ((np.dot(w, np.transpose(x))) + bias) < 0:
            count += 1
            w += alpha * y_train[r] * x
            bias += alpha * y_train[r]
        P_train[iter] = PercentCorrect(X_train, y_train, w, bias)
        P_test[iter] = PercentCorrect(X_test, y_test, w, bias)
        # if P_test[iter] - P_test[iter - 1] > 10:
        #     alpha = alpha/2
        # print(alpha)

    print('Percentage Correct After Training %6.2f %6.2f'
          % (PercentCorrect(X_train, y_train, w, bias), PercentCorrect(X_test, y_test, w, bias)))
    print(bias)
    print(w)
    return MaxIter, P_train, P_test, bias, w


def show_Data_result(NumDataPerClass, m1, m2, w=0, bias=0):
    C = [[2, 1], [1, 2]]

    A = np.linalg.cholesky(C)

    U1 = np.random.randn(NumDataPerClass, 2)
    X1 = U1 @ A.T + m1

    U2 = np.random.randn(NumDataPerClass, 2)
    X2 = U2 @ A.T + m2

    fig, ax = plt.subplots(figsize=(5, 5))

    point_A = np.linspace(-4, 10, 100)
    point_B = (-w[0] * point_A - bias) / w[1]
    ax.plot(point_A, point_B)
    ax.scatter(X1[:, 0], X1[:, 1], c="c", s=4)
    ax.scatter(X2[:, 0], X2[:, 1], c="r", s=4)
    plt.savefig('data_result_advanced.png')

    plt.show()


def generate_Data(NumDataPerClass, m1, m2):
    '''

    generate 100 samples each from two bi-variate Gaussian densities.
    :param NumDataPerClass:
    :return:
    '''

    C = [[2, 1], [1, 2]]

    A = np.linalg.cholesky(C)

    U1 = np.random.randn(NumDataPerClass, 2)
    X1 = U1 @ A.T + m1

    U2 = np.random.randn(NumDataPerClass, 2)
    X2 = U2 @ A.T + m2

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(X1[:, 0], X1[:, 1], c="c", s=4)
    ax.scatter(X2[:, 0], X2[:, 1], c="r", s=4)
    plt.savefig('data.png')

    plt.show()

    # concatenate data from two classes into one array
    X = np.concatenate((X1, X2), axis=0)

    # setting up targets label
    labelPos = np.ones(NumDataPerClass)
    labelNeg = -1.0 * np.ones(NumDataPerClass)
    y = np.concatenate((labelPos, labelNeg))

    # partitioning the data into training and test sets
    rIndex = np.random.permutation(2 * NumDataPerClass)
    Xr = X[rIndex,]  ## why add comma
    # Xs = X[rIndex]
    # print(Xr - Xs)
    yr = y[rIndex]

    X_train = Xr[0:NumDataPerClass]
    y_train = yr[0:NumDataPerClass]
    X_test = Xr[NumDataPerClass:2 * NumDataPerClass]
    y_test = yr[NumDataPerClass:2 * NumDataPerClass]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_test = pd.DataFrame(data=X_test)
    X_train = pd.DataFrame(data=X_train)
    y_test = pd.DataFrame(data=y_test, columns=['class'])
    y_train = pd.DataFrame(data=y_train, columns=['class'])
    Ntrain = NumDataPerClass
    Ntest = NumDataPerClass

    return X_train, y_train, X_test, y_test, Ntrain, Ntest


def data_of_iris(iris_A, iris_B, iris_A_y, iris_B_y):
    # iris_data = np.concatenate((iris_A, iris_B), axis=0)
    iris_data = pd.concat([iris_A, iris_B])
    iris_label = pd.concat(iris_A_y, iris_B_y)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].scatter(iris_A.iloc[:, 0], iris_A.iloc[:, 1], c="c", s=4)
    ax[0].scatter(iris_B.iloc[:, 0], iris_B.iloc[:, 1], c="r", s=4)
    ax[0].set_xlabel("sepal_length", fontsize=16)
    ax[0].set_ylabel("sepal_width", fontsize=16)
    ax[1].scatter(iris_A.iloc[:, 0], iris_A.iloc[:, 1], c="b", s=4)
    ax[1].scatter(iris_B.iloc[:, 0], iris_B.iloc[:, 1], c="g", s=4)
    ax[1].set_xlabel("petal_length", fontsize=16)
    ax[1].set_ylabel("petal_width", fontsize=16)
    plt.savefig('_3.png')
    r_index = np.random.permutation(len(iris_data))
    label = np.unique(iris_data['class'])

    iris_data['class'] = iris_data['class'].replace(to_replace=label[0], value=-1)
    iris_data['class'] = iris_data['class'].replace(to_replace=label[1], value=1)
    iris_data = iris_data.set_index(np.arange(0, len(iris_data)))
    iris_data = iris_data.reindex(r_index)

    train_data = iris_data.iloc[:int(len(iris_data) * 0.8), :-1]
    train_label = iris_data.iloc[:int(len(iris_data) * 0.8), -1]
    test_data = iris_data.iloc[int(len(iris_data) * 0.8):, :-1]
    test_label = iris_data.iloc[int(len(iris_data) * 0.8):, -1]
    return train_data, train_label, test_data, test_label, len(train_data), len(test_data)


def get_iris_data():
    # data from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#
    # raw_data = np.loadtxt('/home/jry/Downloads/bank-full.csv', delimiter=',')
    # raw_data = pd.read_csv('/home/jry/Downloads/bank-full.csv', sep=';')
    # print(raw_data.info)
    raw_url = urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    raw_data = pd.read_csv(raw_url, sep=',', header=None,
                           names=('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'))
    data = raw_data.iloc[:, :-1]
    label = raw_data.iloc[:, -1]
    data = pd.DataFrame(data=data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    label = pd.DataFrame(data=label, columns=['class'])
    return data, label


def get_pca_data():
    raw_url = urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    raw_data = pd.read_csv(raw_url, sep=',', header=None,
                           names=('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'))
    data = raw_data.iloc[:, :-1]
    label = raw_data.iloc[:, -1]
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    data = pd.DataFrame(data=data, columns=['sepal', 'petal'])
    label = pd.DataFrame(data=label, columns=['class'])
    return data, label


def pca_iris(A_data, A_label, B_data, B_label):
    iris_data = pd.concat([A_data, B_data])
    iris_label = pd.concat([A_label, B_label])
    r_index = np.random.permutation(len(iris_data))
    label = np.unique(iris_label)
    iris_label['class'] = iris_label['class'].replace(to_replace=label[0], value=-1)
    iris_label['class'] = iris_label['class'].replace(to_replace=label[1], value=1)
    iris_data = iris_data.set_index(np.arange(0, len(iris_data)))
    iris_data = iris_data.reindex(r_index)
    iris_label = iris_label.set_index(np.arange(0, len(iris_label)))
    iris_label = iris_label.reindex(r_index)
    train_data = iris_data.iloc[:int(len(iris_data) * 0.8)]
    train_label = iris_label.iloc[:int(len(iris_data) * 0.8)]
    test_data = iris_data.iloc[int(len(iris_data) * 0.8):]
    test_label = iris_label.iloc[int(len(iris_data) * 0.8):]

    return train_data, train_label, test_data, test_label, len(train_data), len(test_data)


if __name__ == '__main__':
    # m1 = [[2.5, 2.5]]
    # m2 = [[10, 10]]
    # X_train, y_train, X_test, y_test, Ntrain, Ntest = generate_Data(200, m1, m2)

    data, label = get_iris_data()
    iris_setosa_data = data[:50]
    iris_setosa_label = label[:50]
    iris_versicolor_data = data[50:100]
    iris_versicolor_label = label[50:100]
    iris_vergin_data = data[100:]
    iris_vergin_label = label[100:]
    # generate iris_setosa data / iris_versicolor data
    # X_train, y_train, X_test, y_test, Ntrain, Ntest = data_of_iris(iris_setosa, iris_versicolor)
    # generate iris_setosa_data / iris_vergin data
    # X_train, y_train, X_test, y_test, Ntrain, Ntest = data_of_iris(iris_setosa, iris_vergin)
    # generate iris_versicolor_data / iris_vergin data
    X_train, y_train, X_test, y_test, Ntrain, Ntest = pca_iris(iris_versicolor_data, iris_versicolor_label, iris_vergin_data, iris_vergin_label)


    # data, label = get_pca_data()
    # iris_setosa_x = data[:50]
    # iris_setosa_y = label[:50]
    # iris_versicolor_x = data[50:100]
    # iris_versicolor_y = label[50:100]
    # iris_vergin_x = data[100:]
    # iris_vergin_y = label[100:]
    # # iris_versicolor = raw_data[50:100]
    # # iris_vergin = raw_data[100:]
    #
    # X_train_1, y_train_1, X_test_1, y_test_1, Ntrain_1, Ntest_1 = pca_iris(iris_setosa_x, iris_setosa_y,
    #                                                                        iris_versicolor_x, iris_versicolor_y)
    # X_train_2, y_train_2, X_test_2, y_test_2, Ntrain_2, Ntest_2 = pca_iris(iris_setosa_x, iris_setosa_y, iris_vergin_x,
    #                                                                        iris_vergin_y)
    # X_train_3, y_train_3, X_test_3, y_test_3, Ntrain_3, Ntest_3 = pca_iris(iris_versicolor_x, iris_versicolor_y,
    #                                                                        iris_vergin_x, iris_vergin_y)

    initial_weight = np.random.randn(len(X_train.T))
    MaxIter, P_train, P_test, bias, weight = perceptron_learning(X_train, y_train, X_test, y_test,
                                                                           N_train=Ntrain,
                                                                           w=initial_weight)
    #
    # sklearn model of Percetron
    # model = Perceptron()
    #
    # initial_weight = np.random.randn(len(X_train_1.T))
    # #
    # MaxIter_1, P_train_1, P_test_1, bias_1, weight_1 = perceptron_learning(X_train_1, y_train_1, X_test_1, y_test_1,
    #                                                                        N_train=Ntrain_1,
    #                                                                        w=initial_weight)
    #
    # model.fit(X_train_1, y_train_1)
    # yh_train_1 = model.predict(X_train_1)
    # print("Accuracy on training set:%6.2f" % (accuracy_score(yh_train_1, y_train_1)))
    #
    # yh_test_1 = model.predict(X_test_1)
    # print("Accuracy on test set:%6.2f" % (accuracy_score(yh_test_1, y_test_1)))
    #
    # MaxIter_2, P_train_2, P_test_2, bias_2, weight_2 = perceptron_learning(X_train_2, y_train_2, X_test_2, y_test_2,
    #                                                                        N_train=Ntrain_2,
    #                                                                        w=initial_weight)
    # model.fit(X_train_2, y_train_2)
    # yh_train_2 = model.predict(X_train_2)
    # print("Accuracy on training set:%6.2f" % (accuracy_score(yh_train_2, y_train_2)))
    #
    # yh_test_2 = model.predict(X_test_2)
    # print("Accuracy on test set:%6.2f" % (accuracy_score(yh_test_2, y_test_2)))
    #
    #
    # MaxIter_3, P_train_3, P_test_3, bias_3, weight_3 = perceptron_learning(X_train_3, y_train_3, X_test_3, y_test_3,
    #                                                                        N_train=Ntrain_3,
    #                                                                        w=initial_weight)
    #
    # model.fit(X_train_3, y_train_3)
    # yh_train_3 = model.predict(X_train_3)
    # print("Accuracy on training set:%6.2f" % (accuracy_score(yh_train_3, y_train_3)))
    #
    # yh_test_3 = model.predict(X_test_3)
    # print("Accuracy on test set:%6.2f" % (accuracy_score(yh_test_3, y_test_3)))
    #
    # fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
    # ax[0].set_ylim(-1, 1)
    # ax[1].set_ylim(-1, 1)
    # ax[2].set_ylim(-1, 1)
    # ax[0].scatter(iris_setosa_x['sepal'], iris_setosa_x['petal'], c='r')
    # ax[0].scatter(iris_versicolor_x['sepal'], iris_versicolor_x['petal'], c='g')
    # ax[1].scatter(iris_setosa_x['sepal'], iris_setosa_x['petal'], c='r')
    # ax[1].scatter(iris_vergin_x['sepal'], iris_vergin_x['petal'], c='b')
    # ax[2].scatter(iris_versicolor_x['sepal'], iris_versicolor_x['petal'], c='g')
    # ax[2].scatter(iris_vergin_x['sepal'], iris_vergin_x['petal'], c='b')
    # line_x_1 = np.linspace(-4, 4, 100)
    # line_y_1 = (-weight_1[0] * line_x_1 - bias_1) / weight_1[1]
    # line_y_2 = (-weight_2[0] * line_x_1 - bias_2) / weight_2[1]
    # line_y_3 = (-weight_3[0] * line_x_1 - bias_3) / weight_3[1]
    # ax[0].plot(line_x_1, line_y_1)
    # ax[1].plot(line_x_1, line_y_2)
    # ax[2].plot(line_x_1, line_y_3)
    # plt.savefig('iris_result.png')





    # show_Data_result(200, m1, m2, weight, bias)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(MaxIter), P_train, 'b', label="Training")
    ax.plot(range(MaxIter), P_test, 'r', label="Test")
    ax.grid(True)
    ax.legend()
    ax.set_title('Perceptron Learning')
    ax.set_ylabel('Training and Test Accuracies', fontsize=16)
    ax.set_xlabthel('Iteration', fontsize=16)
    plt.savefig("after_pca.jpg")
    plt.savefig('perception_result_changed.png')
