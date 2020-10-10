#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# plot decision boundary
# manually generate grid and calculate score for every grid

from plot_scatter_label import plot_scatter_label


def predict(theta, x):
    theta = np.mat(theta).T

    if x.shape[1] < theta.shape[0]:
        x = np.hstack((np.ones(x.shape[0]).reshape(-1, 1), x))

    product = x * theta
    return [1 if i >= 0 else -1 for i in product]


def plot_decision_boundary(theta,
                           predict_function=predict,
                           x_min=-10,
                           x_max=10,
                           y_min=-10,
                           y_max=10,
                           step=100,
                           scatter=False):
    # theta = [theta1, theta2]

    grid = np.hstack((np.linspace(x_min, x_max, step).reshape(
        (-1, 1)), np.linspace(y_min, y_max, step).reshape((-1, 1))))

    predicted_grid = np.zeros((step, step))

    # calculates each y by looping through every combination of grid
    for i in range(step):
        for j in range(step):
            predicted_grid[i][j] = sum(
                predict_function([grid[j][0], grid[i][1]]))

    plt.figure(figsize=((8, 8)))
    plt.contourf(grid[:, 0], grid[:, 1], predicted_grid, levels=2, cmap='RdGy')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f'decision boundary for theta = {theta}')
    plt.grid()
    plt.show()


def plot_decision_boundary_mesh(theta,
                                predict_function=predict,
                                x_min=-10,
                                x_max=10,
                                y_min=-10,
                                y_max=10,
                                step=100,
                                scatter=False):

    # mesh grid method
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, step),
                         np.linspace(y_min, y_max, step))

    grid = np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)))

    predicted_grid = predict(theta, grid)
    predicted_grid = np.array(predicted_grid).reshape(xx.shape)

    plt.figure(figsize=((8, 8)))
    plt.contourf(xx, yy, predicted_grid, levels=2, cmap='RdGy')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f'decision boundary for theta = {theta}')
    plt.grid()
    plt.show()


def plot_decision_boundary_mesh_scatter(theta,
                                        X,
                                        Y,
                                        predict_function=predict,
                                        x_min=0,
                                        x_max=100,
                                        y_min=0,
                                        y_max=100,
                                        step=100,
                                        scatter=False):

    print(f'step: {step}')
    # mesh grid method
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, step),
                         np.linspace(y_min, y_max, step))

    grid = np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)))

    predicted_grid = predict(theta, grid)
    predicted_grid = np.array(predicted_grid).reshape(xx.shape)

    plt.figure(figsize=((8, 8)))
    plt.contourf(xx, yy, predicted_grid, levels=2, cmap='RdGy')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f'decision boundary for theta = {theta}')
    plt.grid()

    data = pd.DataFrame(
        np.hstack((X, Y)),
        columns=['offset', 'symmetry', 'intensity_average', 'label'])

    class0 = data[data['label'].isin([-1])]
    class1 = data[data['label'].isin([1])]

    plt.scatter(x=class1['symmetry'],
                y=class1['intensity_average'],
                color='green',
                marker='o',
                label='positive')
    plt.scatter(x=class0['symmetry'],
                y=class0['intensity_average'],
                color='red',
                marker='x',
                label='negative')


if __name__ == "__main__":
    plot_decision_boundry([1, 1], x_start=-10100, y_start=-100)
