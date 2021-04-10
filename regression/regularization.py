from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge, lasso_path, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def lasso_feature_selection(learning_rate, X, target, column_list):
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2)
    l1 = Lasso(alpha=learning_rate, max_iter=10000)
    l1.fit(X_train, y_train)
    th_lasso = l1.predict(X_test)
    print(learning_rate)
    features = list(np.nonzero(l1.coef_))
    ten_top_feature_list = np.argsort(np.absolute(l1.coef_))[::-1][:10]
    x_columns = column_list[ten_top_feature_list]
    error_m = error(y_test, th_lasso)
    print(len(x_columns))
    print('error:{:.3f}'.format(error_m))
    return th_lasso, features[0].size, x_columns, error_m


def error(target, predict):
    return ((target - predict) ** 2).mean()





if __name__ == '__main__':
    file_path = '/home/jry/Downloads/' + 'Husskonen_Solubility_Features.xlsx'
    sol = pd.read_excel(file_path)
    sol_X = sol.iloc[:, 5:]
    sol_X_1 = sol_X[['MLOGP2', 'B07[C-C]', 'B01[C-O]', 'B01[C-Cl]', 'B03[C-N]', 'B02[C-O]',
       'piPC10', 'B02[C-Cl]', 'B04[C-Cl]', 'B05[C-Cl]']]
    column_list = sol_X.columns
    column_list_1 = sol_X_1.columns
    t = sol["LogS.M."].values
    minmax = MinMaxScaler()
    sol_X = minmax.fit_transform(sol_X)
    sol_X_1 = minmax.fit_transform(sol_X_1)
    X_train, X_test, y_train, y_test = train_test_split(sol_X, t, test_size=0.2)
    X_train_1, X_test_1, y_train, y_test = train_test_split(sol_X_1, t, test_size=0.2)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    th_test = lin.predict(X_test)
    th_train = lin.predict(X_train)

    lin = LinearRegression()
    lin.fit(X_train_1, y_train)
    th_test_1 = lin.predict(X_test_1)
    th_train_1 = lin.predict(X_train_1)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
    ax[0].set_xlabel("train_target_before")
    ax[0].set_ylabel("train_predict_before")
    ax[0].set_xlim(-12, 1)
    ax[0].set_ylim(-12, 1)
    ax[0].scatter(y_train, th_train, c='g', s=3)
    ax[1].set_xlabel("test_target_before")
    ax[1].set_ylabel("test_predict_before")
    ax[1].set_xlim(-12, 1)
    ax[1].set_ylim(-12, 1)
    ax[1].scatter(y_test, th_test, c='g', s=3)
    ax[2].set_xlabel("train_target_after")
    ax[2].set_ylabel("train_predict_after")
    ax[2].set_xlim(-12, 1)
    ax[2].set_ylim(-12, 1)
    ax[2].scatter(y_train, th_train_1, c='g', s=3)
    ax[3].set_xlabel("test_target_after")
    ax[3].set_ylabel("test_predict_after")
    ax[3].set_xlim(-12, 1)
    ax[3].set_ylim(-12, 1)
    ax[3].scatter(y_test, th_test_1, c='g', s=3)
    plt.savefig("L2_model_result_end.jpg")
    print(error(y_test, th_test))
    print(error(y_test, th_test_1))
    print(r2_score(y_test, th_test))
    print(r2_score(y_test, th_test_1))
    #
    # l1 = Lasso(alpha=0.2)
    # l1.fit(X_train, y_train)
    # th_test = l1.predict(X_test)
    # th_train = l1.predict(X_train)
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # ax[0].set_xlabel("train_target")
    # ax[0].set_ylabel("train_predict")
    # ax[0].set_xlim(-12, 2)
    # ax[0].set_ylim(-12, 2)
    # ax[0].scatter(y_train, th_train, c='c', s=3)
    # ax[1].set_xlabel("test_target")
    # ax[1].set_ylabel("test_predict")
    # ax[1].set_xlim(-12, 2)
    # ax[1].set_ylim(-12, 2)
    # ax[1].scatter(y_test, th_test, c='g', s=3)
    # plt.savefig("l1_model_result_not.jpg")
    # print(error(y_test, th_test))
    #
    # l2 = Lasso(alpha=0.2)
    # l2.fit(X_train, y_train)
    # th_test = l2.predict(X_test)
    # th_train = l2.predict(X_train)
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # ax[0].set_xlabel("train_target")
    # ax[0].set_ylabel("train_predict")
    # ax[0].set_xlim(-12, 2)
    # ax[0].set_ylim(-12, 2)
    # ax[0].scatter(y_train, th_train, c='c', s=3)
    # ax[1].set_xlabel("test_target")
    # ax[1].set_ylabel("test_predict")
    # ax[1].set_xlim(-12, 2)
    # ax[1].set_ylim(-12, 2)
    # ax[1].scatter(y_test, th_test, c='g', s=3)
    # plt.savefig("l2_model_result_not.jpg")
    # print(error(y_test, th_test))

    # optimization
    # alphas = np.arange(0.01, 2, step=0.01)
    # error_means =[]
    # feature_num_list = []
    # for alpha in alphas:
    #     th_lasso, feature_num, columns_name, error_mean = lasso_feature_selection(alpha, sol_X, t, column_list)
    #     error_means.append(error_mean)
    #     feature_num_list.append(feature_num)
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    # ax.set_xlabel("alpha", fontsize=16)
    # ax.set_ylabel("error_mean", fontsize=16)
    # ax.plot(alphas, error_means)
    # plt.savefig("error_means with alpha.jpg")
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    # ax.set_xlabel("alpha", fontsize=16)
    # ax.set_ylabel("non_zero_feature_num", fontsize=16)
    # ax.plot(alphas, feature_num_list)
    # plt.savefig(" feature_num with alpha.jpg")
    # #
    #
    # # find weight by learning rate
    # learning_rate = 0.2
    # lab_result = []
    # while True:
    #     th_lasso, feature_num, columns_name, error_mean = lasso_feature_selection(learning_rate, sol_X, t, column_list)
    #     lab_result.append([learning_rate, feature_num, error_mean, columns_name])
    #     if feature_num == 10:
    #         break
    #     elif feature_num > 10:
    #         learning_rate += learning_rate * 0.1
    #     else:
    #         learning_rate -= learning_rate * 0.01
    #
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    # ax.scatter(y_test, th_lasso, c='c', s=3)
    # plt.savefig("try.png")
    #
    # print(columns_name)
    # result = pd.DataFrame(data=lab_result, columns=["alpha", "feature_num", "error_mean", "columns_name"])
    # result.to_csv("result_1.csv")

