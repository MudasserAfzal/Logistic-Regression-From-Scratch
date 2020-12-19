##Plot dataset and design boundary
##Credit given https://towardsdatascience.com/decision-boundary-visualization-a-z-6a63ae9cca7d

import matplotlib.pyplot as plt
import numpy as np

def map_features(x, degree):
    x_old = x.copy()
    
    column_index = 0
    for i in range(2, degree+1):
        for j in range(0, i+1):
            itm = np.multiply(x_old[:,0]**(i-j), x_old[:,1]**(j))
            itm = itm.reshape(itm.shape[0], 1)
            x = np.append(x, itm, axis = 1)
            column_index+=1
    return x
	
	
def plotData(ann_model, fs, X_train, y_train, xlabel = "X1", ylabel="X2"):
    '''
    ann_model: the model obtained through training
    fs: the noramlize function for feature scaling
    '''
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


    if ann_model is not None:
        h = .02  # step size in the mesh
        x_min, x_max = min(X_train[:, 0]), max(X_train[:, 0])
        y_min, y_max = min(X_train[:, 1]), max(X_train[:, 1])
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        nX = fs(map_features(np.c_[xx.ravel(), yy.ravel()], 6).T)

        # Put the result into a color plot
        Z = ann_model.predict(nX)
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)