#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_decision_boundary import plot_decision_boundary_mesh_scatter

#%%

S = np.array([[1, 2, 1], [-1, 2, -1], [0, -1, -1]])

X = S[:, :-1]
Y = S[:, -1:]

print(S)

print(X)
print(Y)


#%%
def predict(theta, x):
    theta = np.mat(theta).T
    product = x * theta
    return [1 if i >= 0 else -1 for i in product]


def classify_prediction(y, theta, x):
    # y(theta*x)
    # if <=0, wrong prediction
    # if >0, right prediction
    h = 1 if x @ theta >= 0 else -1

    return h == y
    # return sum(x @ theta * y) > 0


def calculate_training_error(theta, X, Y):
    error = 0
    for t in range(X.shape[0]):
        if not classify_prediction(Y[t], theta, X[t]):
            error += 1
    return error


#%%
theta = np.array([1, -.8])

# plot_decision_boundary_mesh_scatter(theta,
#                                     X,
#                                     Y,
#                                     x_min=-10,
#                                     y_min=-10,
#                                     x_max=10,
#                                     y_max=10,
#                                     step=200)

print(f"initial training error: {calculate_training_error(theta, X, Y)}\n")

#%%
t = 0
k = 0
while True:
    if classify_prediction(Y[t % X.shape[0]], theta, X[t % X.shape[0]]):
        pass
    else:
        print(f'{theta} + {Y[t % X.shape[0]] * X[t % X.shape[0]]}')
        theta = theta + Y[t % X.shape[0]] * X[t % X.shape[0]]

    t += 1

    # plot_decision_boundary_mesh_scatter(theta,
    #                                     X,
    #                                     Y,
    #                                     x_min=-10,
    #                                     y_min=-10,
    #                                     x_max=10,
    #                                     y_max=10,
    #                                     step=200)
    print(t)
    print(theta)
    print(calculate_training_error(theta, X, Y))

    if calculate_training_error(theta, X, Y) == 0:
        break
