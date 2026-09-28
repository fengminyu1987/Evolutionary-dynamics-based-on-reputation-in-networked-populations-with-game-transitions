"""
author: Yuji Zhang
time: 2023-12-20
setting: 0-cooperation-0, 1-defection
description: cooperation frequency varies with time in WS
"""

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def init_graph():
    graph = nx.random_graphs.watts_strogatz_graph(size, 4, 0.3)
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
    own_fitness = nodes_fitness[node]
    random_neighbor_fitness = nodes_fitness[random_neighbor]
    own_reputation = nodes_reputation[node]
    random_neighbor_reputation = nodes_reputation[random_neighbor]
    return (1 - delta) / (1 + np.exp((own_fitness - random_neighbor_fitness) / kappa)) + delta / (
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


def game_transition(transition_type):
    if transition_type == "deterministic state independent":
        for (e, s) in zip(edges_states.keys(), edges_states.values()):
            if s == 0:
                if strategies[e[0]] == 0 and strategies[e[1]] == 0:
                    edges_states[e] = 0
                else:
                    edges_states[e] = 1
            else:
                if strategies[e[0]] == 0 and strategies[e[1]] == 0:
                    edges_states[e] = 0
                else:
                    edges_states[e] = 1
    else:
        pass


size = 1600
a = 4
kappa = 1
state_space = [0, 1]
state_b = {0: 6, 1: 3}
c = 1
delta = 0.2
time_steps = 2000
simulations = 50
transition_modes = ["deterministic state independent", "no game transition"]

# create folders to save results
first_folder = r"evolutionary time = " + str(time_steps)
if not os.path.exists(first_folder):
    os.mkdir(first_folder)
second_folder = first_folder + "\\" + "b = " + str(list(state_b.values()))
if not os.path.exists(second_folder):
    os.mkdir(second_folder)
target_path = second_folder + "\\"

# start simulation
for transition_mode in transition_modes:
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
                game_transition(transition_mode)
                last_round_strategies = strategies.copy()
        total_cooperator_rates += cooperator_rates
    average_cooperator_rates = total_cooperator_rates / simulations
    plt.plot(range(time_steps), average_cooperator_rates, label=transition_mode)
    average_cooperator_rates = pd.DataFrame(average_cooperator_rates)
    average_cooperator_rates.to_csv(target_path + transition_mode + " t=" + str(time_steps) + " under WS" + ".csv",
                                    index=False, header=False)

plt.legend(loc="best")
plt.show()
