"""
author: Yuji Zhang
time: 2024-03-20
setting: 0-cooperation-0, 1-defection
description: cooperation frequency varies with time among three states in SL
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


def payoff_from_payoff_matrix(strategy1, strategy2, state):
    b = state_b[state]
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


def init_game_state():
    game_state_dictionary = {}
    for edge in G.edges:
        game_state_dictionary[edge] = 1
    return game_state_dictionary


def calculate_payoff_and_fitness():
    fitness_dictionary = {}
    payoff_dictionary = {}
    for n in G.nodes:
        neighbors = list(nx.neighbors(G, n))
        payoff = 0
        own_strategy = strategies[n]
        # in SL in networkx,the first index node of edge is smaller the second one.
        for neighbor in neighbors:
            if n < neighbor:
                s = edges_states[(n, neighbor)]
            else:
                s = edges_states[(neighbor, n)]
            neighbor_strategy = strategies[neighbor]
            payoff += payoff_from_payoff_matrix(own_strategy, neighbor_strategy, s)
        payoff_dictionary[n] = payoff
        fitness_dictionary[n] = (a / np.pi * np.arctan(nodes_reputation[n]) + a / 2) * payoff
    return payoff_dictionary, fitness_dictionary


def game_transition():
    for (e, s) in zip(edges_states.keys(), edges_states.values()):
        if strategies[e[0]] + strategies[e[1]] == 0:
            transition_matrix = np.array([[1 - p, 0, p],
                                          [0, 1 - p, p],
                                          [0, 0, 1]])
            probability_vector = transition_matrix[s]
            edges_states[e] = np.random.choice(state_space, 1, p=probability_vector)[0]
        if strategies[e[0]] + strategies[e[1]] == 1:
            transition_matrix = np.array([[1 - p, p, 0],
                                          [0, 1, 0],
                                          [0, p, 1 - p]])
            probability_vector = transition_matrix[s]
            edges_states[e] = np.random.choice(state_space, 1, p=probability_vector)[0]
        else:  # s==2
            transition_matrix = np.array([[1, 0, 0],
                                          [p, 1 - p, 0],
                                          [p, 0, 1 - p]])
        probability_vector = transition_matrix[s]
        edges_states[e] = np.random.choice(state_space, 1, p=probability_vector)[0]


L = 40
a = 4
kappa = 1
state_space = [0, 1, 2]
state_b = {0: 6, 1: 4, 2: 2}
c = 1
delta = 0.2
time_steps = 2000
simulations = 50

ps = [0.3, 0.4, 0.5, 0.6, 0.7]
# create folders to save results
first_folder = r"evolutionary time = " + str(time_steps)
if not os.path.exists(first_folder):
    os.mkdir(first_folder)
second_folder = first_folder + "\\" + "b = " + str(list(state_b.values()))
if not os.path.exists(second_folder):
    os.mkdir(second_folder)
target_path = second_folder + "\\"

# start simulation
for p in ps:
    total_cooperator_rates = np.zeros(time_steps)
    for simulation in range(simulations):
        print("进行第" + str(simulation + 1) + "轮仿真")
        cooperator_rates = []
        G, strategies, last_round_strategies = init_graph()
        nodes_reputation = init_reputation()
        edges_states = init_game_state()
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
                nodes_payoff, nodes_fitness = calculate_payoff_and_fitness()
                # strategy update
                for node in G.nodes:
                    neighbors = list(nx.neighbors(G, node))
                    random_neighbor = choose_from_neighbours(node)
                    probability = fermi_function()
                    if random.random() < probability:
                        strategies[node] = last_round_strategies[random_neighbor]
                update_reputation()
                game_transition()
                last_round_strategies = strategies.copy()
        total_cooperator_rates += cooperator_rates
    average_cooperator_rates = total_cooperator_rates / simulations
    plt.plot(range(time_steps), average_cooperator_rates,label = "p=" + str(p))
    average_cooperator_rates = pd.DataFrame(average_cooperator_rates)
    average_cooperator_rates.to_csv(target_path + " p=" + str(p) + " under SL" + ".csv",
                                    index=False, header=False)

plt.legend(loc="best")
plt.show()
