"""
author: Yuji Zhang
time: 2023-03-18
setting: 0-cooperation-0, 1-defection
description: cooperation frequency varies with network sizes in SL
"""

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def init_graph():
    graph = nx.grid_graph((L, L), periodic=True)
    strategy_dictionary = {}
    for n in graph.nodes:
        if random.random() < 0.5:
            strategy_dictionary[n] = 0
        else:
            strategy_dictionary[n] = 1
    last_round_strategy_dictionary = strategy_dictionary.copy()
    return graph, strategy_dictionary, last_round_strategy_dictionary


def payoff_from_payoff_matrix(strategy1, strategy2):
    payoff_matrix = [[b - c, -c], [b, 0]]
    return payoff_matrix[strategy1][strategy2]


def calculate_cooperator_rate():
    cooperator_number = list(strategies.values()).count(0)
    return cooperator_number / len(G.nodes)


def calculate_number_of_cooperators_of_neighbours(vertex):
    number = 0
    neighbors = list(nx.neighbors(G, vertex))
    for neighbor in neighbors:
        if strategies[neighbor] == 0:
            number += 1
    return number


def norm_distributions(num):
    if num == 0:
        mu = -0.2
        sigma = 0.1
        return np.random.normal(mu, sigma)
    elif num == 1:
        mu = 0
        sigma = 0.1
        return np.random.normal(mu, sigma)
    elif num == 2:
        mu = 0.1
        sigma = 0.1
        return np.random.normal(mu, sigma)
    else:
        mu = 0.3
        sigma = 0.1
        return np.random.normal(mu, sigma)


def init_reputation():
    reputation_dictionary = {}
    for n in G.nodes:
        reputation_dictionary[n] = np.tan((1 / a - 0.5) * np.pi)
    return reputation_dictionary


def update_reputation():
    for n in G.nodes:
        number_of_cooperators = calculate_number_of_cooperators_of_neighbours(n)
        nodes_reputation[n] += norm_distributions(number_of_cooperators)


def fermi_function():
    own_payoff = nodes_payoff[node]
    random_neighbor_payoff = nodes_payoff[random_neighbor]
    own_reputation = nodes_reputation[node]
    random_neighbor_reputation = nodes_reputation[random_neighbor]
    return (1 - delta) / (1 + np.exp((own_payoff - random_neighbor_payoff) / kappa)) + delta / (
            1 + np.exp((own_reputation - random_neighbor_reputation) / kappa))


# choose from neighbors by fitness
def choose_from_neighbours(n):
    neighbors = list(nx.neighbors(G, n))
    fitness_list = [nodes_fitness[n] for n in neighbors]
    if max(fitness_list) == min(fitness_list):
        return random.choice(neighbors)
    fitness_ratios = [(f - min(fitness_list)) / (max(fitness_list) - min(fitness_list)) for f in fitness_list]
    pros = [f / sum(fitness_ratios) for f in fitness_ratios]
    return neighbors[np.random.choice(len(neighbors), 1, p=pros)[0]]


def calculate_payoff_and_fitness():
    fitness_dictionary = {}
    payoff_dictionary = {}
    for n in G.nodes:
        neighbors = list(nx.neighbors(G, n))
        payoff = 0
        own_strategy = strategies[n]
        # in networkx,the first index node of edge is smaller the second one.
        for neighbor in neighbors:
            neighbor_strategy = strategies[neighbor]
            payoff += payoff_from_payoff_matrix(own_strategy, neighbor_strategy)
        payoff_dictionary[n] = payoff
        fitness_dictionary[n] = (a / np.pi * np.arctan(nodes_reputation[n]) + a / 2) * payoff
    return payoff_dictionary, fitness_dictionary


def standard_brownian_motion():
    points = time_steps + 1
    # parameter of standard brownian motion
    mean, variance = 0, 1.0
    Z = np.random.normal(mean, variance, points)
    interval = [0, time_steps]
    dt = (interval[1] - interval[0]) / (points - 1)
    W = np.zeros(points)
    W[0] = 0
    for idx in range(points - 1):
        real_idx = idx + 1
        W[real_idx] = W[real_idx - 1] + np.sqrt(dt) * Z[idx]
    return W


def geometric_brownian_motion(W):
    points = time_steps + 1
    b = np.zeros(points)
    b[0] = b0
    for idx in range(points - 1):
        real_idx = idx + 1
        b[real_idx] = b[0] * np.exp((mu - 0.5 * (sigma ** 2)) * real_idx + sigma * W[real_idx])
    return b


# parameter of geometric brownian motion
sigma = 0.0001
mu = 0.5 * (sigma ** 2)
b0 = 4
b1 = 4
# parameter of evolutionary game
Ls = [30, 35, 40, 45, 50, 55, 60]
a = 4
kappa = 1
c = 1
delta = 0.2
time_steps = 2000
simulations = 30
transition_mode = "probabilistic behavior independent"

# create folders to save results
first_folder = r"evolutionary time = " + str(time_steps)
if not os.path.exists(first_folder):
    os.mkdir(first_folder)
second_folder = first_folder + "\\" + "b0 = " + str(b0) + " b1 = " + str(b1)
if not os.path.exists(second_folder):
    os.mkdir(second_folder)
target_path = second_folder + "\\"

# start simulation
different_average_c = []
for L in Ls:
    print("网络规模：" + str(L * L))
    total_cooperator_rates = np.zeros(time_steps)
    for simulation in range(simulations):
        # generate one sample path of standard brownian motion
        W = standard_brownian_motion()
        # generate the corresponding geometric brownian motion
        b_list = geometric_brownian_motion(W)
        print("进行第" + str(simulation + 1) + "轮仿真")
        cooperator_rates = []
        G, strategies, last_round_strategies = init_graph()
        nodes_reputation = init_reputation()
        for time_step in range(time_steps):
            cooperator_rate = calculate_cooperator_rate()
            if cooperator_rate == 1:
                # the population has evolved into pure cooperators (no mutation), therefore end this turn evolution
                cooperator_rates += [1 for i in range(time_steps - time_step)]
                break
            elif cooperator_rate == 0:
                # the population has evolved into pure defectors (no mutation), therefore end this turn evolution
                cooperator_rates += [0 for i in range(time_steps - time_step)]
                break
            else:
                # the population has not evolved into a pure state, therefore continue evolution
                cooperator_rates.append(cooperator_rate)
                b = b_list[time_step] + b1
                nodes_payoff, nodes_fitness = calculate_payoff_and_fitness()
                # strategy update
                for node in G.nodes:
                    neighbors = list(nx.neighbors(G, node))
                    random_neighbor = choose_from_neighbours(node)
                    probability = fermi_function()
                    if random.random() < probability:
                        strategies[node] = last_round_strategies[random_neighbor]
                update_reputation()
                last_round_strategies = strategies.copy()
        total_cooperator_rates += cooperator_rates
    average_cooperator_rates = total_cooperator_rates / simulations
    different_average_c.append(np.mean(average_cooperator_rates[-50:]))

plt.plot(Ls, different_average_c, label=transition_mode)
different_average_c = pd.DataFrame(different_average_c)
different_average_c.to_csv(target_path + transition_mode + " under SL.csv", index=False, header=False)
plt.legend(loc="best")
plt.show()
