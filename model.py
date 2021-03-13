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
    
    return graph


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


def update_probabilities_pa(graph, e):
    # updates probabilities (using preferential attachment) for choosing a new vertex - as Π=k/2E
    return dict(zip(graph.keys(), [len(graph[k])/(2*e) for k in sorted(graph.keys())]))


def update_probabilities_ra(graph):
    # updates probabilities (using preferential attachment) for choosing a new vertex - as Π=1/N
    return dict(zip(graph.keys(), [1/len(graph.keys()) for k in graph.keys()]))


def update_probabilities_mixed(graph, e, q=2/3):
    # updates probabilities (using preferential attachment) for choosing a new vertex - as Π=qΠ_pa + (1-q)Π_ra
    space = [0, 1]
    prob = [q, 1-q]
    choice = np.random.choice(space, size=1, p=prob)

    if choice == 0:
        return update_probabilities_pa(graph, e)
    else:
        return update_probabilities_ra(graph)


def add_vertex(graph, probs, m=2, method='pa', q=2/3):
    elements = list(graph.keys())  # nodes to choose from for new vertex
    probabilities = list(probs.values())  # probabilities for each vertex
    j = len(graph)  # index of new vertex - is always the length of the graph
    graph[len(graph)] = set()  # initialise adjacency values of new vertex as empty

    # Choose m new vertices using given method - probability distribution
    new_vertices = np.random.choice(elements, size=m, replace=False, p=probabilities)
    for i in new_vertices:
        add_edges(graph, j, i)  # connect new vertex edges with vertices chosen using probabilities
    
    degrees = update_degrees(graph)
    e = update_e(degrees)
    
    if method == 'pa':  # preferential attachment
        probabilities = update_probabilities_pa(graph, e)
        return graph, e, degrees, probabilities
    elif method == 'ra':  # random attachment
        probabilities = update_probabilities_ra(graph)
        return graph, e, degrees, probabilities
    elif method == 'mixed':  # mixed attachment
        probabilities = update_probabilities_mixed(graph, e, q)
        return graph, e, degrees, probabilities
    else:
        raise ValueError("Not a valid probability method.")


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


def deg_dist_theoretical(k, m=2):
    # theoretical degree distribution
    return (2*m*(m+1))/(k*(k+1)*(k+2))


if __name__ == '__main__':
    pass