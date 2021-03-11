from model import initialise_graph, update_e, update_degrees, update_probabilities_pa, add_vertex, save_graph
import numpy as np
import argparse
import sys
import time
import networkx as nx
import matplotlib.pyplot as plt

"""
Georgios Alevras - 28/03/2021
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1

Additional Dependencies: argparse, sys
--------------------------------------

    This file is run as a script to verify that the BA model works correctly. Run individual tests
    to ensure that the model is sensible and follows the theoretical predictions.
    It contains the 2 following testing methods:
            1. test_adjacency_list: Tests adjacency list for being 'sensible', i.e. no duplicates and symmetry,
                by converting it to an adjacency matrix and checking it.
            2. test_average_degrees: Tests that the average number of degrees in the system is approximately
                twice the value of m used.
"""


def test_adjacency_list():
    """ Makes an adjacancy matrix from the adjacency list to check there are no duplicates and 
    that it is symmetrical - simple, undirected and unweighted graph. """

    load = loading()
    print('\nTesting the adjacency list')

    graph = initialise_graph(size=4, m=2)
    degrees = update_degrees(graph)
    e = update_e(degrees)
    probabilities = update_probabilities_pa(graph, e)
    
    for i in range(int(1e3)):
        graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=2, method='pa')
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
    
    adjacency_matrix_transpose = adjacency_matrix.transpose()

    assert np.array_equal(adjacency_matrix, adjacency_matrix_transpose)

    if np.array_equal(adjacency_matrix, adjacency_matrix_transpose):
        print('\n**************************\nAdjacency List Test PASSED\n**************************\n\n')
    else:
        print('\n**************************\nAdjacency List Test FAILED\n**************************\n\n')


def test_average_degrees():
    """ Ensures that the average degree of a graph is approximately 2m when using preferential attachment """

    load = loading()

    m = 2
    graph = initialise_graph(size=4, m=2)
    degrees = update_degrees(graph)
    e = update_e(degrees)
    probabilities = update_probabilities_pa(graph, e)

    for i in range(int(1e3)):
        graph, e, degrees_2, probabilities = add_vertex(graph, probabilities, m=m)
        sys.stdout.write(next(load))
        sys.stdout.flush()
        sys.stdout.write('\b')

    assert round(np.average(list(degrees_2.values()), 0)) == int(2*m)

    m = 3
    graph = initialise_graph(size=4, m=2)
    degrees = update_degrees(graph)
    e = update_e(degrees)
    probabilities = update_probabilities_pa(graph, e)

    for i in range(int(1e3)):
        graph, e, degrees_3, probabilities = add_vertex(graph, probabilities, m=m, method='pa')
        sys.stdout.write(next(load))
        sys.stdout.flush()
        sys.stdout.write('\b')
    
    assert round(np.average(list(degrees_3.values()), 0)) == int(2*m)

    m = 4
    graph = initialise_graph(size=4, m=2)
    degrees = update_degrees(graph)
    e = update_e(degrees)
    probabilities = update_probabilities_pa(graph, e)

    for i in range(int(1e3)):
        graph, e, degrees_4, probabilities = add_vertex(graph, probabilities, m=m, method='pa')
        sys.stdout.write(next(load))
        sys.stdout.flush()
        sys.stdout.write('\b')
    print(' \r')

    print('\nWith m=' + str(2) + ', Avg k: ', round(np.average(list(degrees_2.values())), 2))
    print('With m=' + str(3) + ', Avg k: ', round(np.average(list(degrees_3.values())), 2))
    print('With m=' + str(4) + ', Avg k: ', round(np.average(list(degrees_4.values())), 2))

    assert round(np.average(list(degrees_4.values()), 0)) == int(2*m)

    print('\n***************************\nAverage Degree Tests PASSED\n***************************\n\n')


def plot_graph():
    """ Plot the graph at each time t to observe it grows as expected """

    load = loading()
    print('\nPlotting graph to check it')

    graph = initialise_graph(size=4, m=2)
    degrees = update_degrees(graph)
    e = update_e(degrees)
    probabilities = update_probabilities_pa(graph, e)
    
    for i in range(int(1e1)):
        graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=2, method='pa')

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


def loading():
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
    else:
        raise ValueError("Not a valid test, please enter a digit between 0 and 2, e.g. -t 2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Complexity & Networks: Networks Project - Testing Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-t', '--test_number', type=float, help='Specify Test Number to perform')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided