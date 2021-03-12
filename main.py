from model import initialise_graph, update_e, update_degrees, update_probabilities_pa, update_probabilities_ra, update_probabilities_mixed, add_vertex, deg_dist_theoretical
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
from sklearn.metrics import r2_score
import scipy.stats as st

"""
Georgios Alevras - 28/03/2021
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1
Matplotlib Version used: 3.3.1
Progress Version used: 1.5
Sklearn Version used: 0.0
Scipy Version: 1.5.2

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
        
        size = int(8e4)
        bar = Bar('Code Running', max=size)
        for i in range(size):
            graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=2, method='pa')
            bar.next()
        bar.finish()      
        
        degrees_x, degrees_y = logbin(list(degrees.values()), scale=1.2)
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
        plt.plot(degrees_x, degrees_y*80000, label=r'$logbin \: data$')
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
        # plt.xlim(1e0, 1e3)
        # plt.ylim(1e-2, 1e5)
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
        # plt.xlim(1e0, 2e2)
        # plt.ylim(1e0, 1e3)
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


def phase_1_task_3(compute=True, plot=False):
    if compute:
        graph_1 = initialise_graph(size=4, m=1)
        degrees_1 = update_degrees(graph_1)
        e_1 = update_e(degrees_1)
        probabilities_1 = update_probabilities_pa(graph_1, e_1)
        
        size = int(1e4)
        bar = Bar('Code Running', max=size)
        for i in range(size):
            graph_1, e_1, degrees_1, probabilities_1 = add_vertex(graph_1, probabilities_1, m=1, method='pa')
            bar.next()
        bar.finish()      
        
        degrees_x_1, degrees_y_1 = logbin(list(degrees_1.values()), scale=1.2)
        degree_dist_1 = Counter(list(degrees_1.values()))
        degrees_1 = list(degree_dist_1.keys())
        occurence_1 = list(degree_dist_1.values())
        
        file = open('Files/Phase1/phase_1_task_3_m1.txt', 'wb')
        pickle.dump(degrees_1, file)
        pickle.dump(occurence_1, file)
        pickle.dump(degrees_x_1, file)
        pickle.dump(degrees_y_1, file)
        file.close()

        graph_2 = initialise_graph(size=4, m=2)
        degrees_2 = update_degrees(graph_2)
        e_2 = update_e(degrees_2)
        probabilities_2 = update_probabilities_pa(graph_2, e_2)
        
        size = int(1e4)
        bar = Bar('Code Running', max=size)
        for i in range(size):
            graph_2, e_2, degrees_2, probabilities_2 = add_vertex(graph_2, probabilities_2, m=2, method='pa')
            bar.next()
        bar.finish()      
        
        degrees_x_2, degrees_y_2 = logbin(list(degrees_2.values()), scale=1.2)
        degree_dist_2 = Counter(list(degrees_2.values()))
        degrees_2 = list(degree_dist_2.keys())
        occurence_2 = list(degree_dist_2.values())
        
        file = open('Files/Phase1/phase_1_task_3_m2.txt', 'wb')
        pickle.dump(degrees_1, file)
        pickle.dump(occurence_2, file)
        pickle.dump(degrees_x_2, file)
        pickle.dump(degrees_y_2, file)
        file.close()

        graph_3 = initialise_graph(size=4, m=3)
        degrees_3 = update_degrees(graph_3)
        e_3 = update_e(degrees_3)
        probabilities_3 = update_probabilities_pa(graph_3, e_3)
        
        size = int(1e4)
        bar = Bar('Code Running', max=size)
        for i in range(size):
            graph_3, e_3, degrees_3, probabilities_3 = add_vertex(graph_3, probabilities_3, m=3, method='pa')
            bar.next()
        bar.finish()      
        
        degrees_x_3, degrees_y_3 = logbin(list(degrees_3.values()), scale=1.2)
        degree_dist_3 = Counter(list(degrees_3.values()))
        degrees_3 = list(degree_dist_3.keys())
        occurence_3 = list(degree_dist_3.values())
        
        file = open('Files/Phase1/phase_1_task_3_m3.txt', 'wb')
        pickle.dump(degrees_3, file)
        pickle.dump(occurence_3, file)
        pickle.dump(degrees_x_3, file)
        pickle.dump(degrees_y_3, file)
        file.close()

        graph_4 = initialise_graph(size=5, m=4)
        degrees_4 = update_degrees(graph_4)
        e_4 = update_e(degrees_4)
        probabilities_4 = update_probabilities_pa(graph_4, e_4)
        
        size = int(1e4)
        bar = Bar('Code Running', max=size)
        for i in range(size):
            graph_4, e_4, degrees_4, probabilities_4 = add_vertex(graph_4, probabilities_4, m=4, method='pa')
            bar.next()
        bar.finish()      
        
        degrees_x_4, degrees_y_4 = logbin(list(degrees_4.values()), scale=1.2)
        degree_dist_4 = Counter(list(degrees_4.values()))
        degrees_4 = list(degree_dist_4.keys())
        occurence_4 = list(degree_dist_4.values())
        
        file = open('Files/Phase1/phase_1_task_3_m4.txt', 'wb')
        pickle.dump(degrees_4, file)
        pickle.dump(occurence_4, file)
        pickle.dump(degrees_x_4, file)
        pickle.dump(degrees_y_4, file)
        file.close()

    if plot:
        file = open('Files/Phase1/phase_1_task_3_m1.txt', 'rb')
        degrees_1 = pickle.load(file)
        occurence_1 = pickle.load(file)
        degrees_x_1 = pickle.load(file)
        degrees_y_1 = pickle.load(file)
        file.close()

        file = open('Files/Phase1/phase_1_task_3_m2.txt', 'rb')
        degrees_2 = pickle.load(file)
        occurence_2 = pickle.load(file)
        degrees_x_2 = pickle.load(file)
        degrees_y_2 = pickle.load(file)
        file.close()

        file = open('Files/Phase1/phase_1_task_3_m3.txt', 'rb')
        degrees_3 = pickle.load(file)
        occurence_3 = pickle.load(file)
        degrees_x_3 = pickle.load(file)
        degrees_y_3 = pickle.load(file)
        file.close()

        file = open('Files/Phase1/phase_1_task_3_m4.txt', 'rb')
        degrees_4 = pickle.load(file)
        occurence_4 = pickle.load(file)
        degrees_x_4 = pickle.load(file)
        degrees_y_4 = pickle.load(file)
        file.close()

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'       
        plt.plot(degrees_x_1, degrees_y_1, 'x', label=r'$m=1, \: logbin \: data$')
        plt.plot(degrees_x_2, degrees_y_2, 'x', label=r'$m=2, \: logbin \: data$')
        plt.plot(degrees_x_3, degrees_y_3, 'x', label=r'$m=3, \: logbin \: data$')
        plt.plot(degrees_x_4, degrees_y_4, 'x', label=r'$m=4, \: logbin \: data$')
        x_space_1 = np.linspace(min(degrees_x_1), max(degrees_x_1), len(degrees_x_1))
        x_space_2 = np.linspace(min(degrees_x_2), max(degrees_x_2), len(degrees_x_2))
        x_space_3 = np.linspace(min(degrees_x_3), max(degrees_x_3), len(degrees_x_3))
        x_space_4 = np.linspace(min(degrees_x_4), max(degrees_x_4), len(degrees_x_4))
        r_2_m_1 = r2_score(degrees_y_1, deg_dist_theoretical(x_space_1, m=1))
        r_2_m_2 = r2_score(degrees_y_2, deg_dist_theoretical(x_space_2, m=2)) 
        r_2_m_3 = r2_score(degrees_y_3, deg_dist_theoretical(x_space_3, m=3))
        r_2_m_4 = r2_score(degrees_y_4, deg_dist_theoretical(x_space_4, m=4))
        print(r_2_m_1, r_2_m_2, r_2_m_3, r_2_m_4)
        plt.plot(x_space_1, deg_dist_theoretical(x_space_1, m=1), '--', label=r'$m=1, \: Theoretical$')
        plt.plot(x_space_2, deg_dist_theoretical(x_space_2, m=2), '--', label=r'$m=2, \: Theoretical$')
        plt.plot(x_space_3, deg_dist_theoretical(x_space_3, m=3), '--', label=r'$m=3, \: Theoretical$')
        plt.plot(x_space_4, deg_dist_theoretical(x_space_4, m=4), '--', label=r'$m=4, \: Theoretical$')
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
        # plt.xlim(1e0, 1e3)
        # plt.ylim(1e-2, 1e5)
        plt.savefig('Plots/phase_1_task_3_m1.png')
        plt.show()


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
        tasks = [1, 2, 3]
        if args.task not in tasks:  # If task No. provided is not valid raise value error
            raise ValueError("Phase No. provided not valid, must be either 1 or 2")
        else:
            if args.task == 1 and args.execute:
                phase_1_task_1(compute=True, plot=False)
            elif args.task == 1 and args.execute == False:
                phase_1_task_1(compute=False, plot=True)
            if args.task == 2 and args.execute:
                phase_1_task_2(compute=True, plot=False)
            elif args.task == 2 and args.execute == False:
                phase_1_task_2(compute=False, plot=True)
            if args.task == 3 and args.execute:
                phase_1_task_3(compute=True, plot=False)
            elif args.task == 3 and args.execute == False:
                phase_1_task_3(compute=False, plot=True)
            else:
                raise ValueError("Does not exist. Please enter -t 0, 1 or 3")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Complexity & Networks: Networks Project - Main Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-p', '--phase_number', type=int, help='Specify Phase Number')
    parser.add_argument('-t', '--task', type=int, help='Task number to be executed')
    parser.add_argument('-e', '--execute', action='store_true', help='Flag: if present will execute rather than plot task')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided