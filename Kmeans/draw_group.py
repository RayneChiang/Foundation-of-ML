from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    # #plot the mean and scatter of the result
    path = os.path.dirname(__file__)
    # data = pd.read_csv(path + "/data.csv", index_col=None)
    # label = pd.read_csv(path + "/sklearn_label.csv", index_col=None)
    # columns = label.columns
    #
    # for column in columns:
    #     data_column = label[column]
    #     label_column = pd.DataFrame(data=data_column.values, columns=['label'])
    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     group_list = np.unique(label_column)
    #     for group in group_list:
    #         index_list = label_column[label_column['label'] == group].index.tolist()
    #         X_n = data.iloc[index_list]
    #         X_axis = X_n['X_axis']
    #         X_mean = np.mean(X_axis)
    #         Y_axis = X_n['Y_axis']
    #         Y_mean = np.mean(Y_axis)
    #         ax.scatter(X_axis, Y_axis, s=2)
    #         ax.plot(X_mean, Y_mean, 'ro')
    #     plt.savefig('sklearn_'+column)


    # show J of iteration
    iteration = pd.read_csv(path + "/iteration_mean.csv", index_col=None)
    # iteration= pd.read_csv(path + "/sklearn_mean.csv", index_col=None)
    X_label = list(range(len(iteration)))
    X_label.remove(0)
    fig, ax = plt.subplots(figsize=(6, 6))
    line_1 = ax.plot(X_label, iteration['J'][1:], marker='o', label='J of K')
    Y_data = len(X_label)*[iteration['J'][0]]
    line_2 = ax.plot(X_label, Y_data, label ='J of origin data')
    ax.legend()
    ax.set_xlabel('K')
    ax.set_ylabel('J (distortion_measure)')
    ax.set_title('manual Result of K-means')
    plt.grid(True)
    plt.savefig('manual Result of K-means')