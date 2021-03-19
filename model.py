import numpy as np
import random as rnd

"""
Georgios Alevras - 28/03/2021
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1

Additional Dependencies: random
-------------------------------

    This file defines the BA model / graph and the methods needed for it. 
    It includes 9 methods:
        1. initialise_graph: creates a graph of a given size and m
        2. add_edges: connects a source vertex to a destination vertex in the graph
        3. update_degrees: updates the dictionary holding the degrees of each vertex
        4. update_e: updates the total number of edges in the graph
        5. update_probabilities_pa: updates the probability of choosing a vertex using Preferntial Attachment
        6. update_probabilities_ra: updates the probability of choosing a vertex using Random Attachment
        7. update_probabilities_mixed: updates the probability of choosing a vertex using Mixed Attachment
        8. add_vertex: adds a new vertex to a graph and connects it to other vertices using a given probability dist
        9. save_graph: saves a graph in a format compatible with networks so that it can be plotted
        10. deg_dist_theoretical: theoretical degree distribution, using Gamma functions (discrete model)
"""


def initialise_graph(size=4, m=2):
    # a dictionary representing the adjacency list
    graph = {x: set() for x in range(size)}  # keys: vertices, values: respective neighbours
    
    for i in range(len(graph)):
        new_nodes = rnd.sample([x for x in range(len(graph)) if x!= i], m)  # connects each vertex to m randomly chosen vertices
        for j in new_nodes:
            add_edges(graph, i, j)  # connects each vertex with its new vertices

    options = []
    for i in range(len(graph)):
        for j in range(len(list(graph[i]))):
            options.append(i)

    return graph, options


def add_edges(graph, source, destination):
    # connects ends of edges with vertices - updating both vertices in dictionary
    graph[source].add(destination)
    graph[destination].add(source)


def update_degrees(graph):
    # updates degrees for each vertex
    return dict(zip(graph.keys(), [len(graph[k]) for k in sorted(graph.keys())]))


def update_e(degrees):
    # updates e - total number of vertices in graph (sum of degrees of all vertices over 2)
    return int(sum(list(degrees.values()))/2)


def add_vertex(graph, options, m=2):
    new_nodes = set()
    while len(new_nodes) < m:
        new_nodes.add(options[rnd.randint(0, len(options)-1)])
    
    j = len(graph)  # index of new vertex - is always the length of the graph
    graph[len(graph)] = set()  # initialise adjacency values of new vertex as empty

    new_nodes = list(new_nodes)
    for i in new_nodes:
        add_edges(graph, j, i)
        options.append(j)
        options.append(i)

    return graph, options


def save_graph(graph, PATH_NAME):
    # Saves a graph in a .txt file so that it can be read and plotted using networkx
    with open(PATH_NAME, 'w') as file:
        nodes = list(graph.keys())
        edges = list(graph.values())

        for i in range(len(graph)):
            word = ""
            for x in edges[i]:
                word += str(x) + ' '
            file.write(str(nodes[i]) + ' ' + str(word)+'\n')


def deg_dist_theoretical_pa(k, m=2):
    # theoretical degree distribution for preferential attachment
    return (2*m*(m+1))/(k*(k+1)*(k+2))


def k_max_pa(N, m):
    return (-1 + np.sqrt(1 + 4*N*m*(m+1))*0.5)


if __name__ == '__main__':
    pass
