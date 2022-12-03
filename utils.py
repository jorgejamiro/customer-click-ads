import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

import i18n


def fig_confustion_matrix(actual_data, predict_data, width, height):
    fig = plt.figure(figsize=(width, height))
    cm = confusion_matrix(actual_data, predict_data)
    sns.heatmap(cm, annot=True, fmt='d')  
    plt.title(i18n.t('title_graph_cm'))

    return fig 


def fig_prediction_areas(x, y, model, width, height):
    x_set, y_set = x, y

    # Create a meshgrid ranging from the minimum to maximum value for both features
    min_x1 = x_set[:,0].min()
    max_x1 = x_set[:,0].max()
    min_x2 = x_set[:,1].min()
    max_x2 = x_set[:,1].max()
    x1, x2 = np.meshgrid(np.arange(min_x1-1, max_x1+1, step=0.01),
    np.arange(min_x2-1, max_x2+1, step=0.01))

    # Run the classifier to predict
    npa_x = np.array([x1.ravel(), x2.ravel()]).T
    x_predict = model.predict(npa_x)

    x_predict_reshaped = x_predict.reshape(x1.shape)

    # 'Contourf' - a way to show a 3D surface on a 2D plane. 
    # Contours will be created splitting the colored regions that represent each different value of Z (predictions 0 or 1 in our case)
    colors_list = ListedColormap(('grey', 'pink'))

    fig = plt.figure(figsize=(width, height))

    plt.contourf(x1,x2, x_predict_reshaped, alpha = 0.4, cmap = colors_list)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    # plot all the actual points
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1], color = colors_list(i), label = j)
        
    plt.title(i18n.t('title_mesh_graph'))
    plt.xlabel(i18n.t('Time Spent on Site'))
    plt.ylabel(i18n.t('Salary'))
    plt.legend()

    return fig


def fig_histogram(df, feature, width, height):
    fig = plt.figure(figsize=(width, height))
    df[feature].hist(bins=20)
    plt.title(i18n.t(f'Histogram of {feature}'))
    plt.xlabel(i18n.t(f'{feature}'))
    plt.ylabel(i18n.t('Count'))

    return fig