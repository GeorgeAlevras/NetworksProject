from model import initialise_graph, update_e, update_degrees, add_vertex, deg_dist_theoretical_pa, save_graph
import numpy as np
import argparse
import sys
import time
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import math
import scipy.stats as st
import pandas as pd

"""
Georgios Alevras - 28/03/2021
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1
Scipy Version used: 1.5.2
Matplotlib Version used: 3.3.1

Additional Dependencies: argparse, sys, time, pickle, collections, math
-----------------------------------------------------------------------

    This file is run as a script to verify that the BA model works correctly. Run individual tests
    to ensure that the model is sensible and follows the theoretical predictions.
    It contains the 2 following testing methods:
            1. test_adjacency_list: Tests adjacency list for being 'sensible', i.e. no duplicates and symmetry,
                by converting it to an adjacency matrix and checking it.
            2. test_average_degrees: Tests that the average number of degrees in the system is approximately
                twice the value of m used.
            3. plot_graph: obtains the time evolution of a random graph for 10 snapshots to check it grows correctly
            4. test_probabilities: obtain the expected probability distribution for a given graph
"""


def test_adjacency_list():
    """ Makes an adjacancy matrix from the adjacency list to check there are no duplicates and 
    that it is symmetrical - simple, undirected and unweighted graph. """

    load = loading()
    print('\nTesting the adjacency list')

    m = 3
    graph, options = initialise_graph(size=m+1, m=m)  # Create complete graph of size = 4 (m=3)
    for i in range(int(1e3)):
        graph, options = add_vertex(graph, options, m=m)
        sys.stdout.write(next(load))
        sys.stdout.flush()
        sys.stdout.write('\b')
    print(' \r')
    
    vertices = list(graph.keys())
    neighbours = list(graph.values())

    adjacency_matrix = np.zeros((len(vertices), len(vertices)))
    for i in range(len(vertices)):
        for j in range(len(vertices)):
            if j in neighbours[i]:
                adjacency_matrix[i][j] = 1
    
    diagonal = np.diagonal(adjacency_matrix)

    adjacency_matrix_transpose = adjacency_matrix.transpose()

    assert np.array_equal(adjacency_matrix, adjacency_matrix_transpose)  # Transpose is equal to original matrix
    assert np.array_equal(diagonal, np.zeros(len(diagonal)))  # Diagonal is 0 for all values, thus no self loops
    assert (adjacency_matrix >= 0).all() and (adjacency_matrix <= 1).all()  # All values are either 0 o 1, unweighted matrix

    if np.array_equal(adjacency_matrix, adjacency_matrix_transpose):
        print('\n**************************\nAdjacency List Test PASSED\n**************************\n\n')
    else:
        print('\n**************************\nAdjacency List Test FAILED\n**************************\n\n')


def test_average_degrees():
    """ Ensures that the average degree of a graph is approximately 2m when using preferential attachment """

    ms = [2, 3, 4, 5]
    degrees_avg = []
    for m in ms:
        graph, options = initialise_graph(size=m+1, m=m)

        for i in range(int(1e3)):
            graph, options = add_vertex(graph, options, m=m)

        degrees = update_degrees(graph)
        degrees_avg.append(round(np.average(list(degrees.values())), 2))
        
        assert round(np.average(list(degrees.values()), 0)) == int(2*m)
        print('\nWith m=' + str(m) + ', Avg k: ', round(np.average(list(degrees.values())), 2))

    print('\n***************************\nAverage Degree Tests PASSED\n***************************\n\n')


def plot_graph():
    """ Plot the graph at each time t to observe it grows as expected """

    load = loading()
    print('\nPlotting graph to check it')

    m = 3
    graph, options = initialise_graph(size=m+1, m=m)
    
    for i in range(int(1e1)):
        graph, options = add_vertex(graph, options, m=m)

        save_graph(graph, 'testing_graph.txt')
        g = nx.read_adjlist('testing_graph.txt')
        plt.figure(1)
        nx.draw_shell(g, with_labels=True, font_weight='bold')
        plt.savefig('Plots/Testing/testing_graph_' + str(i) + '.png')
        plt.show()

        sys.stdout.write(next(load))
        sys.stdout.flush()
        sys.stdout.write('\b')
    print(' \r')


def test_probabilities(compute=False, plot=True):
    """ Ensure that preferential attachment works correctly """

    if compute:
        m = 3
        graph, options = initialise_graph(size=m+1, m=m)  # Create complete graph of size = 4 (m=3)
        
        for i in range(int(16)):
            graph, options = add_vertex(graph, options, m=m)

        degrees = update_degrees(graph)
        e = update_e(degrees)
        probabilities = dict(zip(graph.keys(), [len(graph[k])/(2*e) for k in sorted(graph.keys())]))
        
        save_graph(graph, 'testing_probabilities_graph.txt')
        file = open('testing_probabilities.txt', 'wb')
        pickle.dump(graph, file)
        pickle.dump(options, file)
        pickle.dump(probabilities, file)
        file.close()

    if plot:
        print('\nTesting the PA probabilities')
        load = loading()

        file = open('testing_probabilities.txt', 'rb')
        graph = pickle.load(file)
        options = pickle.load(file)
        probabilities = pickle.load(file)
        file.close()
        
        # Probability of choosing each node based on final graph as produced in lines 129-130
        comparison = np.array(list(probabilities.values()))
        
        big_list = []
        for i in range(100):  # Repeat process 100 times to obtain avg and std
            probabilities_produced = {x: 0 for x in range(len(graph))}
            for i in range(int(100000)):  # Grow graph once with m=1 100,000 times to see which node is chosen
                graph_temp, options_temp = add_vertex(graph, options, m=1)
                graph_use = graph_temp.copy()
                probabilities_produced[list(graph_use[len(graph_use)-1])[0]] += 1  # increment for node chosen
                del graph[len(graph_temp)-1]  # erase changes made to repeat
                del options[len(options)-1]  # erase changes made to repeat
                del options[len(options)-1]  # erase changes made to repeat
                sys.stdout.write(next(load))
                sys.stdout.flush()
                sys.stdout.write('\b')
            big_list.append(probabilities_produced)
        print(' \r')
        
        my_list = pd.DataFrame(big_list)
        avg = dict(my_list.mean())
        err = dict(my_list.std())

        # Divide by size of repetitions to make frequencies into probabilities
        avg_p = np.array(list(avg.values()))/100000
        err_p = np.array(list(err.values()))/100000
        probabilities_produced = np.array(list(probabilities_produced.values()))/100000
        
        if np.array_equal(np.round(comparison, 1), np.round(probabilities_produced, 1)):
            print('\n**************************\nPA Probability Test PASSED\n**************************\n\n')
        else:
            print('\n**************************\nPA Probability Test FAILED\n**************************\n\n')        

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        chi, p_value = st.chisquare(probabilities_produced, comparison)
        plt.bar(list(probabilities.keys()), comparison, fill=False, label=r'$Expected$')
        plt.bar(list(probabilities.keys()), avg_p, yerr=err_p, alpha=0.5, color='cyan', label=r'$Obtained, \:$' + '\n' + r'$\chi^2={}, \: p-value={}$'.format(
            str(round(chi, 6)), str(round(p_value, 4))))
        plt.legend()
        plt.xlabel(r'$\it{Vertex}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{Attachment \: Probability}$', fontname='Times New Roman', fontsize=17)
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        x_int = range(min(list(probabilities.keys())), math.ceil(max(list(probabilities.keys())))+1)
        matplotlib.pyplot.xticks(x_int)
        plt.xlim(-1, 20)
        plt.ylim(0, 0.175)
        plt.savefig('Plots/Testing/comparison_probabilities.png')

        g = nx.read_adjlist('testing_probabilities_graph.txt')
        plt.figure(2)
        nx.draw_shell(g, with_labels=True, font_weight='bold')
        plt.savefig('Plots/Testing/testing_probabilities.png')
        plt.show()


def loading():
    # Loading symbol when running a function on terminal
    while True:
        for c in '|/-\\':
            yield c


def use_args(args):
    if args.test_number == 0:
        test_adjacency_list()
    elif args.test_number == 1:
        test_average_degrees()
    elif args.test_number == 2:
        plot_graph()
    elif args.test_number == 3:
        test_probabilities()
    else:
        raise ValueError("Not a valid test, please enter a digit between 0 and 2, e.g. -t 2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Complexity & Networks: Networks Project - Testing Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-t', '--test_number', type=float, help='Specify Test Number to perform')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided