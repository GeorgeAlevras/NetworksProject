import numpy as np
import argparse
import random as rnd
import time as time


class Graph:
    def __init__(self, size): 
        self.size = size 
        self.adj = {x: set() for x in range(size)}
    

    def __repr__(self):
        return 'Graph:\n======\nsize = '+str(self.size)+'\nvertices = '+str(list(self.adj.keys()))+'\nadjacency list = '+str(self.adj)+'\n'


    def add_edges(self, source, destination):
        self.adj[source].add(destination)
        self.adj[destination].add(source)


def initialise(graph):
    for i in range(graph.size):
        new_nodes = rnd.sample([x for x in range(graph.size) if x!= i], rnd.randint(1, 4))
        for j in new_nodes:
            graph.add_edges(i, j)


def add_vertex(graph):
    j = graph.size  # index of new vertex
    graph.size += 1  # update size of graph
    graph.adj[size-1] = set()  # update adjacency list

    m = rnd.randint(1, 4)  # No. of edges for new vertex
    for i in range(m):
        graph.add_edge(j, )


def find_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not start in self.adj.keys():
            return None
        for node in self.adj[start]:
            if node not in path:
                newpath = find_path(self, node, end, path)
                if newpath: return newpath
        return None


def find_all_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not start in self.adj.keys():
            return []
        paths = []
        for node in self.adj[start]:
            if node not in path:
                newpaths = find_all_paths(self, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths


def find_shortest_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not start in self.adj.keys():
            return None
        shortest = None
        for node in self.adj[start]:
            if node not in path:
                newpath = find_shortest_path(self, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest


def use_args(args):
    """
        :param args: arguments provided by user at terminal / cmd
    """
    
    if args.execute:
        graph = Graph(size=5)
        initialise(graph)
        print(graph)
        for i in range(3):
            add_vertex(graph)
        print(graph)
        
    else:
        print('No argument found.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Complexity & Networks: Networks Project - Script Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-e', '--execute', action='store_true', help='Execute script')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided