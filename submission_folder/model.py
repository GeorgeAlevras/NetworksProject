import numpy as np
import random as rnd

"""
Georgios Alevras - 24/03/2021
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1

Additional Dependencies: random
-------------------------------

    This file defines the BA model / graph and the methods needed for it. 
    It includes 12 methods:
        1. initialise_graph: creates a graph of a given size and m
        2. add_edges: connects a source vertex to a destination vertex in the graph
        3. update_degrees: updates the dictionary holding the degrees of each vertex
        4. update_e: updates the total number of edges in the graph
        5. add_vertex: adds a new vertex to a graph and connects it to other vertices using a given probability dist
        6. save_graph: saves a graph in a format compatible with networks so that it can be plotted
        7. deg_dist_theoretical_pa: theoretical degree distribution for preferential attachment
        8. deg_dist_theoretical_ra: theoretical degree distribution for random attachment
        9. deg_dist_theoretical_mi_2_3: theoretical degree distribution for mixed preferential attachment for q=2/3
        10. deg_dist_theoretical_mi_1_2: theoretical degree distribution for mixed preferential attachment for q=1/2
        11. k_max_pa: theoretical expected largest degree for preferential attachment
        12. k_max_ra: theoretical expected largest degree for random attachment
"""


def initialise_graph(size=4, m=2):
    # Dictionary representing the adjacency list
    graph = {x: set() for x in range(size)}  # keys: vertices, values: respective neighbours
    
    for i in range(len(graph)):
        new_nodes = rnd.sample([x for x in range(len(graph)) if x!= i], m)  # connects each vertex to all other vertices
        for j in new_nodes:
            add_edges(graph, i, j)  # connects each vertex with its new vertices

    options = []  # List holds all stubs to use for choosing a vetex when using preferential attachment
    for i in range(len(graph)):
        for j in range(len(list(graph[i]))):
            options.append(i)

    return graph, options


def add_edges(graph, source, destination):
    # connects ends of edges with vertices - updating both vertices in dictionary
    graph[source].add(destination)
    graph[destination].add(source)


def update_degrees(graph):
    # Updates / gets degrees for each vertex
    return dict(zip(graph.keys(), [len(graph[k]) for k in sorted(graph.keys())]))


def update_e(degrees):
    # Updates / gets e - total No. of vertices in graph (sum of degrees of all vertices over 2)
    return int(sum(list(degrees.values()))/2)


def add_vertex(graph, options, m=2, method='pa', q=2/3):
    new_nodes = set()  # use of a set (datastructure) to have unique elements (vertice)
    while len(new_nodes) < m:  # Keep adding unique vertices until m are chosen
        if method == 'pa':
            # Choose an index randomly from a list weighted by presence of stubs (degrees) for each vertex
            new_nodes.add(options[rnd.randint(0, len(options)-1)])
        elif method == 'ra':
            # Choose randomly any vertex in graph
            new_nodes.add(rnd.randint(0, len(graph)-1))
        elif method == 'mi':
            # Create a probability space 0: PA, 1: RA to choose from 
            if q == 2/3:
                probs = [0, 0, 1]
            elif q == 1/2:
                probs = [0, 1]
            else:
                raise ValueError("Invalid value of q. Model designed only for q=2/3 and q=1/2.")
            
            choice = probs[rnd.randint(0, len(probs)-1)]
            if choice == 0:  # If 0 use same logic for PA
                new_nodes.add(options[rnd.randint(0, len(options)-1)])
            else:  # If 1 use same logic for RA
                new_nodes.add(rnd.randint(0, len(graph)-1))
        else:
            raise ValueError("Not valid method. Please provide one of: pa, ra, mi")
    
    j = len(graph)  # index of new vertex - always the length of the graph
    graph[len(graph)] = set()  # initialise adjacency values of new vertex as empty

    new_nodes = list(new_nodes)  # List of all new chosen vertices to attach
    for i in new_nodes:
        add_edges(graph, j, i)  # Connect all new vertices
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
    # Theoretical degree distribution for preferential attachment
    return (2*m*(m+1))/(k*(k+1)*(k+2))


def deg_dist_theoretical_ra(k, m=2):
    # Theoretical degree distribution for random attachment
    return ((m/(m+1))**(k-m))*(1/(1+m))


def deg_dist_theoretical_mi_2_3(k, m=2):
    # Theoretical degree distribution for mixed preferential attachment, q=2/3
    return (12*m*(2*m+1)*(2*m+2)*(2*m+3)) / ((k+m)*(k+m+1)*(k+m+2)*(k+m+3)*(6+4*m))


def deg_dist_theoretical_mi_1_2(k, m=2):
    # Theoretical degree distribution for mixed preferential attachment, q=1/2
    return (12*m*(3*m+1)*(3*m+2)*(3*m+3)) / ((k+2*m)*(k+2*m+1)*(k+2*m+2)*(k+2*m+3)*(k+2*m+4))


def k_max_pa(N, m):
    # Theoretical expected largest degree for preferential attachment
    return -1 + 0.5*np.sqrt(1+4*N*m*(m+1))


def k_max_ra(N, m):
    # Theoretical expected largest degree for random attachment
    return m + np.log(N)/np.log((1+m)/m)


if __name__ == '__main__':
    pass
