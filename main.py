from model import initialise_graph, update_e, update_degrees, update_probabilities_pa, update_probabilities_ra, \
    update_probabilities_mixed, add_vertex, deg_dist_theoretical
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
        
        size = int(1e5)
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
        
        size = int(5e4)
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
        
        size = int(5e4)
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
        
        size = int(5e4)
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
        
        size = int(5e4)
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
        x_space_1 = np.linspace(min(degrees_x_1), max(degrees_x_1), 1000)
        x_space_2 = np.linspace(min(degrees_x_2), max(degrees_x_2), 1000)
        x_space_3 = np.linspace(min(degrees_x_3), max(degrees_x_3), 1000)
        x_space_4 = np.linspace(min(degrees_x_4), max(degrees_x_4), 1000)
        r_2_m_1 = r2_score(deg_dist_theoretical(degrees_x_1, m=1), degrees_y_1)
        r_2_m_2 = r2_score(deg_dist_theoretical(degrees_x_2, m=2), degrees_y_2)
        r_2_m_3 = r2_score(deg_dist_theoretical(degrees_x_3, m=3), degrees_y_3)
        r_2_m_4 = r2_score(deg_dist_theoretical(degrees_x_4, m=4), degrees_y_4)
        chi_m1 = st.chisquare(deg_dist_theoretical(degrees_x_1, m=1), degrees_y_1)
        chi_m2 = st.chisquare(deg_dist_theoretical(degrees_x_2, m=2), degrees_y_2)
        chi_m3 = st.chisquare(deg_dist_theoretical(degrees_x_3, m=3), degrees_y_3)
        chi_m4 = st.chisquare(deg_dist_theoretical(degrees_x_4, m=4), degrees_y_4)
        print('R2 values (m: 1, 4): ', r_2_m_1, r_2_m_2, r_2_m_3, r_2_m_4)
        print('Chi_2 values (m: 1, 4): ', chi_m1, chi_m2, chi_m3, chi_m4)
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
        plt.savefig('Plots/phase_1_task_3_i.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'      
        cum_m1 = np.cumsum(degrees_y_1)
        cum_m2 = np.cumsum(degrees_y_2)
        cum_m3 = np.cumsum(degrees_y_3)
        cum_m4 = np.cumsum(degrees_y_4)
        cum_exp_m1 = np.cumsum(deg_dist_theoretical(degrees_x_1, m=1))
        cum_exp_m2 = np.cumsum(deg_dist_theoretical(degrees_x_2, m=2))
        cum_exp_m3 = np.cumsum(deg_dist_theoretical(degrees_x_3, m=3))
        cum_exp_m4 = np.cumsum(deg_dist_theoretical(degrees_x_4, m=4))
        ks_m1 = st.kstest(cum_m1, cum_exp_m1)
        ks_m2 = st.kstest(cum_m2, cum_exp_m2)
        ks_m3 = st.kstest(cum_m3, cum_exp_m3)
        ks_m4 = st.kstest(cum_m4, cum_exp_m4)
        print('KS Test: (m: 1, 4): ', ks_m1, ks_m2, ks_m3, ks_m4)
        plt.plot(degrees_x_1, cum_m1, 'o', label=r'$m=1 \: Obtained \ : Data$')
        plt.plot(degrees_x_2, cum_m2, 'o', label=r'$m=2 \: Obtained \ : Data$')
        plt.plot(degrees_x_3, cum_m3, 'o', label=r'$m=3 \: Obtained \ : Data$')
        plt.plot(degrees_x_4, cum_m4, 'o', label=r'$m=4 \: Obtained \ : Data$')
        plt.plot(degrees_x_1, cum_exp_m1, label=r'$m=1 \: Theoretical \: Data$')
        plt.plot(degrees_x_2, cum_exp_m2, label=r'$m=2 \: Theoretical \: Data$')
        plt.plot(degrees_x_3, cum_exp_m3, label=r'$m=3 \: Theoretical \: Data$')
        plt.plot(degrees_x_4, cum_exp_m4, label=r'$m=4 \: Theoretical \: Data$')
        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{c(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/phase_1_task_3_ii.png')
        
        plt.show()


def phase_1_task_4(compute=True, plot=False):
    if compute:
        
        max_k_1024 = 0
        degree_dicts_1024 = []
        sum_counter = Counter()
        bar = Bar('Code Running', max=5)
        for i in range(5):
            graph = initialise_graph(size=5, m=4)
            degrees = update_degrees(graph)
            e = update_e(degrees)
            probabilities = update_probabilities_pa(graph, e)
            
            size = int(1024)
            for i in range(size):
                graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=4, method='pa')  
            
            degree_dist = Counter(list(degrees.values()))
            degree_dicts_1024.append(degree_dist)
            
            sum_counter += degree_dist

            if max(degree_dist.keys()) > max_k_1024:
                max_k_1024 = max(degree_dist.keys())
            
            bar.next()
        bar.finish()

        degrees_1024 = list(sum_counter.keys())
        occurence_1024 = list(np.array(list(sum_counter.values()))/5)
        degrees_x_1024, degrees_y_1024 = logbin(occurence_1024, scale=1.2)
        file = open('Files/Phase1/phase_1_task_4_1024.txt', 'wb')
        pickle.dump(degrees_1024, file)
        pickle.dump(occurence_1024, file)
        pickle.dump(degrees_x_1024, file)
        pickle.dump(degrees_y_1024, file)
        file.close()
        
        max_k_2048 = 0
        degree_dicts_2048 = []
        sum_counter = Counter()
        bar = Bar('Code Running', max=5)
        for i in range(5):
            graph = initialise_graph(size=5, m=4)
            degrees = update_degrees(graph)
            e = update_e(degrees)
            probabilities = update_probabilities_pa(graph, e)
            
            size = int(2048)
            for i in range(size):
                graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=4, method='pa')  
            
            degree_dist = Counter(list(degrees.values()))
            degree_dicts_2048.append(degree_dist)
            
            sum_counter += degree_dist

            if max(degree_dist.keys()) > max_k_2048:
                max_k_2048 = max(degree_dist.keys())
            
            bar.next()
        bar.finish()

        degrees_2048 = list(sum_counter.keys())
        occurence_2048 = list(np.array(list(sum_counter.values()))/5)
        degrees_x_2048, degrees_y_2048 = logbin(occurence_2048, scale=1.2)
        file = open('Files/Phase1/phase_1_task_4_2048.txt', 'wb')
        pickle.dump(degrees_2048, file)
        pickle.dump(occurence_2048, file)
        pickle.dump(degrees_x_2048, file)
        pickle.dump(degrees_y_2048, file)
        file.close()

        max_k_4096 = 0
        degree_dicts_4096 = []
        sum_counter = Counter()
        bar = Bar('Code Running', max=5)
        for i in range(5):
            graph = initialise_graph(size=5, m=4)
            degrees = update_degrees(graph)
            e = update_e(degrees)
            probabilities = update_probabilities_pa(graph, e)
            
            size = int(4096)
            for i in range(size):
                graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=4, method='pa')  
            
            degree_dist = Counter(list(degrees.values()))
            degree_dicts_4096.append(degree_dist)
            
            sum_counter += degree_dist
            
            if max(degree_dist.keys()) > max_k_4096:
                max_k_4096 = max(degree_dist.keys())
            
            bar.next()
        bar.finish()

        degrees_4096 = list(sum_counter.keys())
        occurence_4096 = list(np.array(list(sum_counter.values()))/5)
        degrees_x_4096, degrees_y_4096 = logbin(occurence_4096, scale=1.2)
        file = open('Files/Phase1/phase_1_task_4_4096.txt', 'wb')
        pickle.dump(degrees_4096, file)
        pickle.dump(occurence_4096, file)
        pickle.dump(degrees_x_4096, file)
        pickle.dump(degrees_y_4096, file)
        file.close()

        max_k_8192 = 0
        degree_dicts_8192 = []
        sum_counter = Counter()
        bar = Bar('Code Running', max=5)
        for i in range(5):
            graph = initialise_graph(size=5, m=4)
            degrees = update_degrees(graph)
            e = update_e(degrees)
            probabilities = update_probabilities_pa(graph, e)
            
            size = int(8192)
            for i in range(size):
                graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=4, method='pa')  
            
            degree_dist = Counter(list(degrees.values()))
            degree_dicts_8192.append(degree_dist)
            
            sum_counter += degree_dist

            if max(degree_dist.keys()) > max_k_8192:
                max_k_8192 = max(degree_dist.keys())
            
            bar.next()
        bar.finish()

        degrees_8192 = list(sum_counter.keys())
        occurence_8192 = list(np.array(list(sum_counter.values()))/5)
        degrees_x_8192, degrees_y_8192 = logbin(occurence_8192, scale=1.2)
        file = open('Files/Phase1/phase_1_task_4_8192.txt', 'wb')
        pickle.dump(degrees_8192, file)
        pickle.dump(occurence_8192, file)
        pickle.dump(degrees_x_8192, file)
        pickle.dump(degrees_y_8192, file)
        file.close()

        max_k_16384 = 0
        degree_dicts_16384 = []
        sum_counter = Counter()
        bar = Bar('Code Running', max=5)
        for i in range(5):
            graph = initialise_graph(size=5, m=4)
            degrees = update_degrees(graph)
            e = update_e(degrees)
            probabilities = update_probabilities_pa(graph, e)
            
            size = int(16384)
            for i in range(size):
                graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=4, method='pa')  
            
            degree_dist = Counter(list(degrees.values()))
            degree_dicts_16384.append(degree_dist)
            
            sum_counter += degree_dist

            if max(degree_dist.keys()) > max_k_16384:
                max_k_16384 = max(degree_dist.keys())
            
            bar.next()
        bar.finish()

        degrees_16384 = list(sum_counter.keys())
        occurence_16384 = list(np.array(list(sum_counter.values()))/5)
        degrees_x_16384, degrees_y_16384 = logbin(occurence_16384, scale=1.2)
        file = open('Files/Phase1/phase_1_task_4_16384.txt', 'wb')
        pickle.dump(degrees_16384, file)
        pickle.dump(occurence_16384, file)
        pickle.dump(degrees_x_16384, file)
        pickle.dump(degrees_y_16384, file)
        file.close()

        max_k_32768 = 0
        degree_dicts_32768 = []
        sum_counter = Counter()
        bar = Bar('Code Running', max=5)
        for i in range(5):
            graph = initialise_graph(size=5, m=4)
            degrees = update_degrees(graph)
            e = update_e(degrees)
            probabilities = update_probabilities_pa(graph, e)
            
            size = int(32768)
            for i in range(size):
                graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=4, method='pa')  
            
            degree_dist = Counter(list(degrees.values()))
            degree_dicts_32768.append(degree_dist)
            
            sum_counter += degree_dist

            if max(degree_dist.keys()) > max_k_32768:
                max_k_32768 = max(degree_dist.keys())
            
            bar.next()
        bar.finish()

        degrees_32768 = list(sum_counter.keys())
        occurence_32768 = list(np.array(list(sum_counter.values()))/5)
        degrees_x_32768, degrees_y_32768 = logbin(occurence_32768, scale=1.2)
        file = open('Files/Phase1/phase_1_task_4_32768.txt', 'wb')
        pickle.dump(degrees_32768, file)
        pickle.dump(occurence_32768, file)
        pickle.dump(degrees_x_32768, file)
        pickle.dump(degrees_y_32768, file)
        file.close()

    if plot:
        file = open('Files/Phase1/phase_1_task_4_1024.txt', 'rb')
        degrees_1024 = pickle.load(file)
        occurence_1024 = pickle.load(file)
        degrees_x_1024 = pickle.load(file)
        degrees_y_1024 = pickle.load(file)
        file.close()

        file = open('Files/Phase1/phase_1_task_4_2048.txt', 'rb')
        degrees_2048 = pickle.load(file)
        occurence_2048 = pickle.load(file)
        degrees_x_2048 = pickle.load(file)
        degrees_y_2048 = pickle.load(file)
        file.close()

        file = open('Files/Phase1/phase_1_task_4_4096.txt', 'rb')
        degrees_4096 = pickle.load(file)
        occurence_4096 = pickle.load(file)
        degrees_x_4096 = pickle.load(file)
        degrees_y_4096 = pickle.load(file)
        file.close()

        file = open('Files/Phase1/phase_1_task_4_8192.txt', 'rb')
        degrees_8192 = pickle.load(file)
        occurence_8192 = pickle.load(file)
        degrees_x_8192 = pickle.load(file)
        degrees_y_8192 = pickle.load(file)
        file.close()

        file = open('Files/Phase1/phase_1_task_4_16384.txt', 'rb')
        degrees_16384 = pickle.load(file)
        occurence_16384 = pickle.load(file)
        degrees_x_16384 = pickle.load(file)
        degrees_y_16384 = pickle.load(file)
        file.close()

        file = open('Files/Phase1/phase_1_task_4_32768.txt', 'rb')
        degrees_32768 = pickle.load(file)
        occurence_32768 = pickle.load(file)
        degrees_x_32768 = pickle.load(file)
        degrees_y_32768 = pickle.load(file)
        file.close()

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'       
        plt.plot(degrees_x_1024, degrees_y_1024, label=r'$N=1024, \: logbin \: data$')
        plt.plot(degrees_x_2048, degrees_y_2048, label=r'$N=2048, \: logbin \: data$')
        plt.plot(degrees_x_4096, degrees_y_4096, label=r'$N=4096, \: logbin \: data$')
        plt.plot(degrees_x_8192, degrees_y_8192, label=r'$N=8192, \: logbin \: data$')
        plt.plot(degrees_x_16384, degrees_y_16384, label=r'$N=16384, \: logbin \: data$')
        plt.plot(degrees_x_32768, degrees_y_32768, label=r'$N=32768, \: logbin \: data$')
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
        plt.savefig('Plots/phase_1_task_4.png')
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
        tasks = [1, 2, 3, 4]
        if args.task not in tasks:  # If task No. provided is not valid raise value error
            raise ValueError("Phase No. provided not valid, must be either 1 or 2")
        else:
            if args.task == 1 and args.execute:
                phase_1_task_1(compute=True, plot=False)
            elif args.task == 1 and args.execute == False:
                phase_1_task_1(compute=False, plot=True)
            elif args.task == 2 and args.execute:
                phase_1_task_2(compute=True, plot=False)
            elif args.task == 2 and args.execute == False:
                phase_1_task_2(compute=False, plot=True)
            elif args.task == 3 and args.execute:
                phase_1_task_3(compute=True, plot=False)
            elif args.task == 3 and args.execute == False:
                phase_1_task_3(compute=False, plot=True)
            elif args.task == 4 and args.execute:
                phase_1_task_4(compute=True, plot=False)
            elif args.task == 4 and args.execute == False:
                phase_1_task_4(compute=False, plot=True)
            else:
                raise ValueError("Does not exist. Please enter -t 1, 2, 3 or 4")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Complexity & Networks: Networks Project - Main Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-p', '--phase_number', type=int, help='Specify Phase Number')
    parser.add_argument('-t', '--task', type=int, help='Task number to be executed')
    parser.add_argument('-e', '--execute', action='store_true', help='Flag: if present will execute rather than plot task')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided