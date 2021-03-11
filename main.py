from model import initialise_graph, update_e, update_degrees, update_probabilities_pa, update_probabilities_ra, update_probabilities_mixed, add_vertex
import argparse
import pickle
import numpy as np
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from collections import Counter
from logbin2020 import logbin
from progress.bar import Bar

"""
Georgios Alevras - 28/03/2021
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1
Matplotlib Version used: 3.3.1
Progress Version used: 1.5

Additional Dependencies: argparse, pickle, time, sys, collections
-----------------------------------------------------------------

    This is the main file where all tasks of the project are run from.
"""


def phase_1_task_1(compute=True, plot=False):
    if compute:
        graph = initialise_graph(size=4, m=2)
        degrees = update_degrees(graph)
        e = update_e(degrees)
        probabilities = update_probabilities_pa(graph, e)
        
        size = int(1e5)
        bar = Bar('Code Running', max=size)
        for i in range(size):
            graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=2, method='pa')
            bar.next()
        bar.finish()      
        
        degrees_x, degrees_y = logbin(list(degrees.values()), scale=1.1)
        degree_dist = Counter(list(degrees.values()))
        degrees = list(degree_dist.keys())
        occurence = list(degree_dist.values())
        
        file = open('Files/Phase1/phase_1_task_1.txt', 'wb')
        pickle.dump(degrees, file)
        pickle.dump(occurence, file)
        pickle.dump(degrees_x, file)
        pickle.dump(degrees_y, file)
        file.close()

    if plot:
        file = open('Files/Phase1/phase_1_task_1.txt', 'rb')
        degrees = pickle.load(file)
        occurence = pickle.load(file)
        degrees_x = pickle.load(file)
        degrees_y = pickle.load(file)
        file.close()

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(degrees, occurence, 'o', label=r'$data$')
        plt.plot(degrees_x, degrees_y*100000, label=r'$logbin \: data$')
        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 1e3)
        plt.ylim(1e-2, 1e5)
        plt.savefig('Plots/phase_1_task_1_i.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        rank = np.linspace(1, len(degrees), len(degrees))
        plt.plot(rank, sorted(degrees, reverse=True), 'o', label=r'$data$')
        plt.legend()
        plt.xlabel(r'$\it{rank}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 2e2)
        plt.ylim(1e0, 1e3)
        plt.savefig('Plots/phase_1_task_1_ii.png')
        plt.show()


def phase_1_task_2(compute=True, plot=False):
    if compute:
        graph = initialise_graph(size=4, m=2)
        degrees = update_degrees(graph)
        e = update_e(degrees)
        probabilities = update_probabilities_ra(graph)

        size = int(1e3)
        bar = Bar('Code Running', max=size)
        for i in range(size):
            graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=2, method='ra')
            bar.next()
        bar.finish()

    if plot:
        pass


def loading():
    while True:
        for c in '|/-\\':
            yield c


def use_args(args):
    phases = [1, 2, 3]  # Valid phases - part of project script

    if args.phase_number is None or args.task is None or args.execute is None:
        raise KeyError("Must provide 3 arguments: phase no. task and execute flag, e.g. -p 1 -t 2 -e")
    
    if args.phase_number not in phases:  # If Phase No. provided is not valid raise value error
        raise ValueError("Phase No. provided not valid, must be either 1, 2 or 3")
    elif args.phase_number == 1:
        tasks = [1, 2]
        if args.task not in tasks:  # If task No. provided is not valid raise value error
            raise ValueError("Phase No. provided not valid, must be either 1 or 2")
        else:
            if args.task == 1 and args.execute:
                phase_1_task_1(compute=True, plot=False)
            elif args.task == 1 and args.execute == False:
                phase_1_task_1(compute=False, plot=True)
            if args.task == 2 and args.execute:
                phase_1_task_2(compute=True, plot=False)
            elif args.task == 2 and not args.execute == False:
                phase_1_task_2(compute=False, plot=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Complexity & Networks: Networks Project - Main Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-p', '--phase_number', type=int, help='Specify Phase Number')
    parser.add_argument('-t', '--task', type=int, help='Task number to be executed')
    parser.add_argument('-e', '--execute', action='store_true', help='Flag: if present will execute rather than plot task')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided