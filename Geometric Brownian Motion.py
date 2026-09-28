import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def standard_brownian_motion():
    points = time_step + 1
    mu, sigma = 0, 1.0
    Z = np.random.normal(mu, sigma, points)
    interval = [0, time_step]
    dt = (interval[1] - interval[0]) / (points - 1)
    t_axis = np.linspace(interval[0], interval[1], points)
    print("dt = " + str(dt))
    print("t_axis=", t_axis)
    W = np.zeros(points)
    W[0] = 0
    for idx in range(points - 1):
        real_idx = idx + 1
        W[real_idx] = W[real_idx - 1] + np.sqrt(dt) * Z[idx]
    return W


def geometric_brownian_motion(W):
    points = time_step + 1
    b = np.zeros(points)
    b[0] = b0
    for idx in range(points - 1):
        real_idx = idx + 1
        b[real_idx] = b[0] * np.exp((mu - 0.5 * (sigma ** 2)) * real_idx + sigma * W[real_idx])
    return b


sigma = 0.0001
mu = 0.5 * (sigma ** 2)
b0 = 3

for i in range(3):
    time_step = 2000
    W = standard_brownian_motion()
    b = geometric_brownian_motion(W)
    plt.plot(range(len(b)), b + b0, label='sample path ' + str(i + 1))

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'style': 'italic',
        'size': 20,
        }

plt.xlabel('$t$', fontdict=font)
plt.ylabel('$B(t)$', fontdict=font)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.show()
