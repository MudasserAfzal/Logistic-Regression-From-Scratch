import numpy as np
import matplotlib.pyplot as plt

def plotData(X_train, y_train, xlabel = "X1", ylabel="X2", w=[], b = 0):
    c0 = c1 = 0 # Counter of label 0 and label 1 instances
    X_train = X_train.T #Transpose input data
    X_train = X_train[:, [0,1]]
    y_train = y_train.T
    for i in range(0, X_train.shape[0]):
        if y_train[i] == 0:
            c0 = c0 + 1
        else:
            c1 = c1 + 1

    x0 = np.ones((c0,2)) # matrix label 0 instances
    x1 = np.ones((c1,2)) # matrix label 1 instances
    k0 = k1 = 0

    for i in range(0,y_train.shape[0]):
        if y_train[i] == 0:
            x0[k0] = X_train[i]
            k0 = k0 + 1
        else:
            x1[k1] = X_train[i]
            k1 = k1 + 1

    X_col = [x0, x1]
    colors = ["green", "blue"] # colours for Scatter Plot


    for x, c in zip(X_col, colors):
        if c == "green":
            plt.scatter(x[:,0], x[:,1], color = c, label = "Negative")
        else:
            plt.scatter(x[:,0], x[:,1], color = c, label = "Positive")

    #Plot decision boundary for Part I
    if(len(w) >0):
        w = w[[0,1], :]
        w = w.reshape(2)
        w = np.insert(w, 0, b)
        # getting the x co-ordinates
        plot_x = np.array([min(X_train[:,0]), max(X_train[:,0])])
        # getting corresponding y co-ordinates of the decision boundary
        plot_y = (-1/w[2]) * (w[1] * plot_x + w[0])
        # Plotting the Single Line Decision Boundary
        plt.plot(plot_x, plot_y, label = "Decision_Boundary")
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)