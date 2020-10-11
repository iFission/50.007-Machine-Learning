# Alex W
# 1003474

# 2a
# theta = [1.78157138, 3.2447227 ]
# error = 0.5812605752543938

# 2b
# batch
# theta = [0.27032973, 0.24127422]
# error = 11.18724893007848

# stochastic
# theta = [0.00156322, 0.00122478]
# error = 13.771874611911194

# 2c
# error = 0.570084436469135
# on order 14:
# 14 0.5497941299008209
# 15 0.5513486786377605

#%%
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_X = pd.read_csv("HW1_data/2/hw1x.dat", header=None, dtype=np.float64)
train_X_with_offset = pd.read_csv("HW1_data/2/hw1x.dat",
                                  header=None,
                                  dtype=np.float64)
train_X_with_offset.insert(1, 'offset', np.ones(len(train_X)))
train_Y = pd.read_csv("HW1_data/2/hw1y.dat", header=None, dtype=np.float64)

train_X.head()


#%%
def get_theta_closed_form(X, Y):
    n = len(X)
    A = 1 / n * X.T @ X
    b = 1 / n * X.T @ Y

    # theta = np.mat(A).I @ b
    theta = np.linalg.pinv(A) @ b

    return theta.T[0]


def create_poly_features(X, degree):
    x1 = X.iloc[:, :1]

    for i in range(2, degree + 1):
        name = f'x^{i}'
        X.insert(X.shape[1], name, np.power(x1, i))

    return X


def plot_scatter_label_and_line(x, y, theta, xlabel, ylabel):
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs {xlabel}')
    plt.grid(True)

    x = np.linspace(0, 2, 100)
    x_with_offset = pd.DataFrame(x)
    x_with_offset.insert(1, 'offset', np.ones(len(x)))

    print(theta.shape[0])
    x_with_offset = create_poly_features(x_with_offset, theta.shape[0] - 1)

    print(x_with_offset.values.shape)
    y = np.mat(x_with_offset.values) * np.mat(theta).T
    plt.plot(x, y, 'r')


def calculate_least_squared_error(X, Y, theta):
    X = np.mat(X)
    Y = np.mat(Y)
    theta = np.mat(theta).T
    predicted_y = X @ theta
    error = predicted_y - Y
    squared_error = np.power(error, 2)
    sum_squared_error = np.sum(squared_error)
    squared_error = sum_squared_error / (2 * X.shape[0])

    return squared_error


def gradient_descent(X, y, theta, alpha, iters):
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta).T
    for _ in range(iters):
        # error = (X * theta) - y
        # gradient = X.T * error # transpose X features, to solve for matrix vector product, X is 100 * n, error is 100*1
        # increment = alpha / X.shape[0] * gradient
        # theta = theta-increment
        theta = theta + alpha / X.shape[0] * (X.T * (y - (X * theta)))
    return np.array(theta.T)[0]


#%%
theta = get_theta_closed_form(train_X_with_offset.values, train_Y.values)
print(theta)
plot_scatter_label_and_line(train_X, train_Y, theta, "X", "Y")

training_error = calculate_least_squared_error(train_X_with_offset.values,
                                               train_Y.values, theta)
print(training_error)

#%%

alpha = 0.01
iters = 5
theta = np.zeros(train_X_with_offset.shape[1])

# perform gradient descent
theta = gradient_descent(train_X_with_offset.values, train_Y.values, theta,
                         alpha, iters)
print(theta)

training_error = calculate_least_squared_error(train_X_with_offset.values,
                                               train_Y.values, theta)
print(training_error)

#%%
import random
random.seed(6)


def stochastic_gradient_descent(X, y, theta, alpha, iters):
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta).T
    for _ in range(iters):

        # choose a random sample
        t = random.randrange(X.shape[0])

        # error = (X * theta) - y
        # gradient = X.T * error # transpose X features, to solve for matrix vector product, X is 100 * n, error is 100*1
        # increment = alpha / X.shape[0] * gradient
        # theta = theta-increment
        theta = theta + alpha / X.shape[0] * (X[t].T * (y[t] - (X[t] * theta)))
    return np.array(theta.T)[0]


#%%

alpha = 0.01
iters = 5
theta = np.zeros(train_X_with_offset.shape[1])

# perform gradient descent
theta = stochastic_gradient_descent(train_X_with_offset.values, train_Y.values,
                                    theta, alpha, iters)
print(theta)

training_error = calculate_least_squared_error(train_X_with_offset.values,
                                               train_Y.values, theta)
print(training_error)

#%%

train_X_with_offset_degree_3 = create_poly_features(
    train_X_with_offset.copy(deep=True), 3)

theta = get_theta_closed_form(train_X_with_offset_degree_3.values,
                              train_Y.values)
print(theta)

plot_scatter_label_and_line(train_X, train_Y, theta, "X", "Y")

#%%
training_error = calculate_least_squared_error(
    train_X_with_offset_degree_3.values, train_Y.values, theta)
print(training_error)

#%%

degree_ls = []
training_error_ls = []

for degree in range(3, 15 + 1):
    train_X_with_offset_degree_n = create_poly_features(
        train_X_with_offset.copy(deep=True), degree)

    theta = get_theta_closed_form(train_X_with_offset_degree_n.values,
                                  train_Y.values)

    training_error = calculate_least_squared_error(
        train_X_with_offset_degree_n.values, train_Y.values, theta)
    print(degree, training_error)

    degree_ls.append(degree)
    training_error_ls.append(training_error)
# %%
plt.plot(degree_ls, training_error_ls)