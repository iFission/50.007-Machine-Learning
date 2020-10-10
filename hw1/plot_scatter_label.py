#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_list = np.array([i for i in range(10)])
y_list = x_list**2


def plot_scatter_label(x, y, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs {xlabel}')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_scatter_label(x_list, y_list, "x", "y")