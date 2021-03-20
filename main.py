from model import initialise_graph, update_e, update_degrees, add_vertex, deg_dist_theoretical_pa, save_graph, \
    k_max_pa, deg_dist_theoretical_ra, k_max_ra, deg_dist_theoretical_mi_2_3, deg_dist_theoretical_mi_1_2
import argparse
import pickle
import numpy as np
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
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
Scipy Version used: 1.5.2
logbin2020 - Referece: Max Falkenberg McGillivray, (2019), Complexity & Networks course

Additional Dependencies: argparse, pickle, time, sys, collections
-----------------------------------------------------------------

    This file runs all tasks for all phases as defined in the project script. 
    It includes 12 methods:
        1. phase_1_task_1: 
        2. phase_1_task_3: 
        3. phase_1_task_4: 
        4. phase_2_task_1: 
        5. phase_2_task_3: 
        6. phase_2_task_4: 
        7. phase_3_task_1: 
        8. phase_3_task_3: 
        9. phase_3_task_4: 
        10. combine_log_bins: 
        11. loading: 
        12. use_args: 

"""


def phase_1_task_1(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e4)
    if compute:
        for m in ms:
            graph, options = initialise_graph(size=m+1, m=m)
            
            bar = Bar('Code Running', max=size)
            for i in range(size):
                graph, options = add_vertex(graph, options, m=m)
                bar.next()
            bar.finish()
            
            degrees = update_degrees(graph)

            degree_dist = Counter(list(degrees.values()))
            degrees = list(degree_dist.keys())
            occurence = list(degree_dist.values())
            
            file = open('Files/Phase1/phase_1_task_1_m'+str(m)+'.txt', 'wb')
            pickle.dump(degrees, file)
            pickle.dump(occurence, file)
            file.close()
    
    if plot:
        degrees = {}
        occurence = {}
        for m in ms:
            file = open('Files/Phase1/phase_1_task_1_m'+str(m)+'.txt', 'rb')
            degrees[m] = pickle.load(file)
            occurence[m] = pickle.load(file)
            file.close()

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(degrees[2], np.array(occurence[2])/size, '.', label=r'$m=2$', color='k')
        plt.plot(degrees[4], np.array(occurence[4])/size, '.', label=r'$m=4$', color='b')
        plt.plot(degrees[8], np.array(occurence[8])/size, '.', label=r'$m=8$', color='r')
        plt.plot(degrees[16], np.array(occurence[16])/size, '.', label=r'$m=16$', color='g')
        plt.plot(degrees[32], np.array(occurence[32])/size, '.', label=r'$m=32$', color='c')
        plt.plot(degrees[64], np.array(occurence[64])/size, '.', label=r'$m=64$', color='m')
        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_pa(x_space[0], m=2), '--', color='k')
        plt.plot(x_space[1], deg_dist_theoretical_pa(x_space[1], m=4), '--', color='b')
        plt.plot(x_space[2], deg_dist_theoretical_pa(x_space[2], m=8), '--', color='r')
        plt.plot(x_space[3], deg_dist_theoretical_pa(x_space[3], m=16), '--', color='g')
        plt.plot(x_space[4], deg_dist_theoretical_pa(x_space[4], m=32), '--', color='c')
        plt.plot(x_space[5], deg_dist_theoretical_pa(x_space[5], m=64), '--', color='m')
        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 2e3)
        plt.ylim(1e-4, 1e0)
        plt.savefig('Plots/phase_1_task_1_i.png')
        plt.show()


def phase_1_task_3(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e4)
    repetitions = [10*2**(6-n) for n in range(0, 6)]

    if compute:
        bar = Bar('Code Running', max=int(size * sum(repetitions)))

        for i, m in enumerate(ms):
            master_array = []
            big_x = []
            big_y = []

            for _ in range(repetitions[i]):
                graph, options = initialise_graph(size=(m+1), m=m)

                for _ in range(size):
                    graph, options = add_vertex(graph, options, m=m)
                    bar.next()

                degrees = update_degrees(graph)
                master_array.append(list(degrees.values()))
                x, y = logbin(list(degrees.values()), scale=1.2)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.2)
            errors = combine_log_bins(big_x, big_y)

            file = open('Files/Phase1/phase_1_task_3_m'+str(m)+'.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        bar.finish()

    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for m in ms:
            file = open('Files/Phase1/phase_1_task_3_m'+str(m)+'.txt', 'rb')
            degrees[m] = pickle.load(file)
            dists[m] = pickle.load(file)
            errors[m] = pickle.load(file)
            file.close()

        r_sq = [r2_score(deg_dist_theoretical_pa(degrees[m], m=m), dists[m]) for m in ms]
        
        chi_sq = []
        for m in ms:
            observed = size*np.array(dists[m])
            indices = np.argwhere(observed<5)
            indices = np.ndarray.flatten(indices)
            observed = observed[observed>=5]
            theoretical = np.delete(size*deg_dist_theoretical_pa(degrees[m], m=m), indices)
            chi_sq.append(st.chisquare(observed, theoretical))
        
        ks_values = [st.ks_2samp(deg_dist_theoretical_pa(degrees[m], m=m), dists[m]) for m in ms]

        print('R2 values: ', r_sq)
        print('Chi_2 values: ', chi_sq)
        print('KS Test values: ', ks_values)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'  

        plt.errorbar(degrees[2], dists[2], yerr=errors[2], marker = '.', ls = ' ', color='k', label=r'$m=2$')
        plt.errorbar(degrees[4], dists[4], yerr=errors[4], marker = '.', ls = ' ', color='b', label=r'$m=4$')
        plt.errorbar(degrees[8], dists[8], yerr=errors[8], marker = '.', ls = ' ', color='r', label=r'$m=8$')
        plt.errorbar(degrees[16], dists[16], yerr=errors[16], marker = '.', ls = ' ', color='g', label=r'$m=16$')
        plt.errorbar(degrees[32], dists[32], yerr=errors[32], marker = '.', ls = ' ', color='c', label=r'$m=32$')
        plt.errorbar(degrees[64], dists[64], yerr=errors[64], marker = '.', ls = ' ', color='m', label=r'$m=64$')

        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_pa(x_space[0], m=2), '--', color='k')
        plt.plot(x_space[1], deg_dist_theoretical_pa(x_space[1], m=4), '--', color='b')
        plt.plot(x_space[2], deg_dist_theoretical_pa(x_space[2], m=8), '--', color='r')
        plt.plot(x_space[3], deg_dist_theoretical_pa(x_space[3], m=16), '--', color='g')
        plt.plot(x_space[4], deg_dist_theoretical_pa(x_space[4], m=32), '--', color='c')
        plt.plot(x_space[5], deg_dist_theoretical_pa(x_space[5], m=64), '--', color='m')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        # plt.xlim(1e0, 1e3)
        # plt.ylim(1e-2, 1e5)
        plt.savefig('Plots/phase_1_task_3_i.png')

        plt.show()


def phase_1_task_4(compute=True, plot=False):
    m = 4
    N = [10**n for n in range(2, 6)]
    repetitions = [10**(5-n) for n in range(1, 5)]

    if compute:
        bar = Bar('Code Running', max=int(np.dot(np.array(N), np.array(repetitions))))

        k_max = []
        k_err = []
        for i, n in enumerate(N):
            master_array = []
            big_x = []
            big_y = []
            
            k_s = []
            for r in range(repetitions[i]):
                graph, options = initialise_graph(size=(m+1), m=m)

                for _ in range(n):
                    graph, options = add_vertex(graph, options, m=m)    
                    bar.next()

                degrees = update_degrees(graph)
                
                k_s.append(max(list(degrees.values())))
                
                master_array.append(list(degrees.values()))
                x, y = logbin(list(degrees.values()), scale=1.2)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.2)
            errors = combine_log_bins(big_x, big_y)
            k_max.append(np.average(k_s))
            k_err.append(np.std(k_s)/np.sqrt(len(k_s)))

            file = open('Files/Phase1/phase_1_task_4_m3_N'+str(n)+'.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        file = open('Files/Phase1/phase_1_task_4_k.txt', 'wb')
        pickle.dump(k_max, file)
        pickle.dump(k_err, file)
        file.close()

    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for n in N:
            file = open('Files/Phase1/phase_1_task_4_m3_N'+str(n)+'.txt', 'rb')
            degrees[n] = pickle.load(file)
            dists[n] = pickle.load(file)
            errors[n] = pickle.load(file)
            file.close()
        file = open('Files/Phase1/phase_1_task_4_k.txt', 'rb')
        k_max = pickle.load(file)
        k_err = pickle.load(file)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'       
        plt.errorbar(degrees[100], dists[100],  yerr=errors[100], marker = '.', ls = ' ', color='b', label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000], dists[1000],  yerr=errors[1000], marker = '.', ls = ' ', color='g', label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000], dists[10000],  yerr=errors[10000], marker = '.', ls = ' ', color='r', label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000], dists[100000], yerr=errors[100000], marker = '.', ls = ' ', color='c', label=r'$Data: \: N=100000$')

        x_space = np.linspace(min(degrees[100]), max(degrees[100000]), 1000)
        plt.plot(x_space, deg_dist_theoretical_pa(x_space, m=4), '--', color='k', label=r'$Theoretical \: Data: \: m=4$')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/phase_1_task_4_i.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.errorbar(N, k_max,  yerr=k_err, marker = '.', ls = ' ', color='b', label=r'$Data$')

        n_space = np.linspace(N[0], N[len(N)-1], 1000)
        plt.plot(n_space, k_max_pa(n_space, m=4), '--', color='k', label=r'$Theoretical$')

        fit_phase, cov_phase = np.polyfit(np.log(N), np.log(k_max), 1, cov=True)
        plt.plot(n_space, np.exp(fit_phase[0]*np.log(n_space) + fit_phase[1]), label=r'$Linear \: Fit$')

        plt.legend()
        plt.xlabel(r'$\it{N}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{k_1}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/phase_1_task_4_ii.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.errorbar(degrees[100]/k_max[0], dists[100]/deg_dist_theoretical_pa(degrees[100], m=4),  yerr=errors[100]/deg_dist_theoretical_pa(degrees[100], m=4), marker = '.', ls = ' ', color='b', label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000]/k_max[1], dists[1000]/deg_dist_theoretical_pa(degrees[1000], m=4),  yerr=errors[1000]/deg_dist_theoretical_pa(degrees[1000], m=4), marker = '.', ls = ' ', color='g', label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000]/k_max[2], dists[10000]/deg_dist_theoretical_pa(degrees[10000], m=4),  yerr=errors[10000]/deg_dist_theoretical_pa(degrees[10000], m=4), marker = '.', ls = ' ', color='r', label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000]/k_max[3], dists[100000]/deg_dist_theoretical_pa(degrees[100000], m=4), yerr=errors[100000]/deg_dist_theoretical_pa(degrees[100000], m=4), marker = '.', ls = ' ', color='c', label=r'$Data: \: N=100000$')        
        plt.legend()
        plt.xlabel(r'$\it{k/k_1}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k) \: / \: p_{\infty}(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/phase_1_task_4_iii.png')

        plt.show()


def phase_2_task_1(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e4)
    if compute:
        for m in ms:
            graph, options = initialise_graph(size=m+1, m=m)
            
            bar = Bar('Code Running', max=size)
            for i in range(size):
                graph, options = add_vertex(graph, options, m=m, method='ra')
                bar.next()
            bar.finish()
            
            degrees = update_degrees(graph)

            degree_dist = Counter(list(degrees.values()))
            degrees = list(degree_dist.keys())
            occurence = list(degree_dist.values())
            
            file = open('Files/Phase2/phase_2_task_1_m'+str(m)+'.txt', 'wb')
            pickle.dump(degrees, file)
            pickle.dump(occurence, file)
            file.close()

    if plot:
        degrees = {}
        occurence = {}
        for m in ms:
            file = open('Files/Phase2/phase_2_task_1_m'+str(m)+'.txt', 'rb')
            degrees[m] = pickle.load(file)
            occurence[m] = pickle.load(file)
            file.close()

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(degrees[2], np.array(occurence[2])/size, '.', label=r'$m=2$', color='k')
        plt.plot(degrees[4], np.array(occurence[4])/size, '.', label=r'$m=4$', color='b')
        plt.plot(degrees[8], np.array(occurence[8])/size, '.', label=r'$m=8$', color='r')
        plt.plot(degrees[16], np.array(occurence[16])/size, '.', label=r'$m=16$', color='g')
        plt.plot(degrees[32], np.array(occurence[32])/size, '.', label=r'$m=32$', color='c')
        plt.plot(degrees[64], np.array(occurence[64])/size, '.', label=r'$m=64$', color='m')
        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_ra(x_space[0], m=2), '--', color='k')
        plt.plot(x_space[1], deg_dist_theoretical_ra(x_space[1], m=4), '--', color='b')
        plt.plot(x_space[2], deg_dist_theoretical_ra(x_space[2], m=8), '--', color='r')
        plt.plot(x_space[3], deg_dist_theoretical_ra(x_space[3], m=16), '--', color='g')
        plt.plot(x_space[4], deg_dist_theoretical_ra(x_space[4], m=32), '--', color='c')
        plt.plot(x_space[5], deg_dist_theoretical_ra(x_space[5], m=64), '--', color='m')
        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        # plt.xlim(1e0, 2e3)
        # plt.ylim(1e-4, 1e0)
        plt.savefig('Plots/phase_2_task_1_i.png')
        plt.show()


def phase_2_task_3(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e4)
    repetitions = [10*2**(6-n) for n in range(0, 6)]

    if compute:
        bar = Bar('Code Running', max=int(size * sum(repetitions)))

        for i, m in enumerate(ms):
            master_array = []
            big_x = []
            big_y = []

            for _ in range(repetitions[i]):
                graph, options = initialise_graph(size=(m+1), m=m)

                for _ in range(size):
                    graph, options = add_vertex(graph, options, m=m, method='ra')
                    bar.next()

                degrees = update_degrees(graph)
                master_array.append(list(degrees.values()))
                x, y = logbin(list(degrees.values()), scale=1.2)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.2)
            errors = combine_log_bins(big_x, big_y)

            file = open('Files/Phase2/phase_2_task_3_m'+str(m)+'.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        bar.finish()

    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for m in ms:
            file = open('Files/Phase2/phase_2_task_3_m'+str(m)+'.txt', 'rb')
            degrees[m] = pickle.load(file)
            dists[m] = pickle.load(file)
            errors[m] = pickle.load(file)
            file.close()

        r_sq = [r2_score(deg_dist_theoretical_ra(degrees[m], m=m), dists[m]) for m in ms]
        
        chi_sq = []
        for m in ms:
            observed = size*np.array(dists[m])
            indices = np.argwhere(observed<5)
            indices = np.ndarray.flatten(indices)
            observed = observed[observed>=5]
            theoretical = np.delete(size*deg_dist_theoretical_ra(degrees[m], m=m), indices)
            chi_sq.append(st.chisquare(observed, theoretical))
        
        ks_values = [st.ks_2samp(deg_dist_theoretical_ra(degrees[m], m=m), dists[m]) for m in ms]

        print('R2 values: ', r_sq)
        print('Chi_2 values: ', chi_sq)
        print('KS Test values: ', ks_values)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'  

        plt.errorbar(degrees[2], dists[2], yerr=errors[2], marker = '.', ls = ' ', color='k', label=r'$m=2$')
        plt.errorbar(degrees[4], dists[4], yerr=errors[4], marker = '.', ls = ' ', color='b', label=r'$m=4$')
        plt.errorbar(degrees[8], dists[8], yerr=errors[8], marker = '.', ls = ' ', color='r', label=r'$m=8$')
        plt.errorbar(degrees[16], dists[16], yerr=errors[16], marker = '.', ls = ' ', color='g', label=r'$m=16$')
        plt.errorbar(degrees[32], dists[32], yerr=errors[32], marker = '.', ls = ' ', color='c', label=r'$m=32$')
        plt.errorbar(degrees[64], dists[64], yerr=errors[64], marker = '.', ls = ' ', color='m', label=r'$m=64$')

        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_ra(x_space[0], m=2), '--', color='k')
        plt.plot(x_space[1], deg_dist_theoretical_ra(x_space[1], m=4), '--', color='b')
        plt.plot(x_space[2], deg_dist_theoretical_ra(x_space[2], m=8), '--', color='r')
        plt.plot(x_space[3], deg_dist_theoretical_ra(x_space[3], m=16), '--', color='g')
        plt.plot(x_space[4], deg_dist_theoretical_ra(x_space[4], m=32), '--', color='c')
        plt.plot(x_space[5], deg_dist_theoretical_ra(x_space[5], m=64), '--', color='m')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        # plt.xlim(1e0, 1e3)
        # plt.ylim(1e-2, 1e5)
        plt.savefig('Plots/phase_2_task_3_i.png')

        plt.show()


def phase_2_task_4(compute=True, plot=False):
    m = 4
    N = [10**n for n in range(2, 6)]
    repetitions = [10**(5-n) for n in range(1, 5)]

    if compute:
        bar = Bar('Code Running', max=int(np.dot(np.array(N), np.array(repetitions))))

        k_max = []
        k_err = []
        for i, n in enumerate(N):
            master_array = []
            big_x = []
            big_y = []
            
            k_s = []
            for r in range(repetitions[i]):
                graph, options = initialise_graph(size=(m+1), m=m)

                for _ in range(n):
                    graph, options = add_vertex(graph, options, m=m, method='ra')    
                    bar.next()

                degrees = update_degrees(graph)
                
                k_s.append(max(list(degrees.values())))
                
                master_array.append(list(degrees.values()))
                x, y = logbin(list(degrees.values()), scale=1.2)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.2)
            errors = combine_log_bins(big_x, big_y)
            k_max.append(np.average(k_s))
            k_err.append(np.std(k_s)/np.sqrt(len(k_s)))

            file = open('Files/Phase2/phase_2_task_4_m3_N'+str(n)+'.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        file = open('Files/Phase2/phase_2_task_4_k.txt', 'wb')
        pickle.dump(k_max, file)
        pickle.dump(k_err, file)
        file.close()


    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for n in N:
            file = open('Files/Phase2/phase_2_task_4_m3_N'+str(n)+'.txt', 'rb')
            degrees[n] = pickle.load(file)
            dists[n] = pickle.load(file)
            errors[n] = pickle.load(file)
            file.close()
        file = open('Files/Phase2/phase_2_task_4_k.txt', 'rb')
        k_max = pickle.load(file)
        k_err = pickle.load(file)
        

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'       
        plt.errorbar(degrees[100], dists[100],  yerr=errors[100], marker = '.', ls = ' ', color='b', label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000], dists[1000],  yerr=errors[1000], marker = '.', ls = ' ', color='g', label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000], dists[10000],  yerr=errors[10000], marker = '.', ls = ' ', color='r', label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000], dists[100000], yerr=errors[100000], marker = '.', ls = ' ', color='c', label=r'$Data: \: N=100000$')

        x_space = np.linspace(min(degrees[100]), max(degrees[100000]), 1000)
        plt.plot(x_space, deg_dist_theoretical_ra(x_space, m=4), '--', color='k', label=r'$Theoretical \: Data: \: m=4$')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/phase_2_task_4_i.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.errorbar(N, k_max,  yerr=k_err, marker = '.', ls = ' ', color='b', label=r'$Data$')

        n_space = np.linspace(N[0], N[len(N)-1], 1000)
        plt.plot(n_space, k_max_ra(n_space, m=4), '--', color='k', label=r'$Theoretical$')

        plt.legend()
        plt.xlabel(r'$\it{N}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{k_1}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_off()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        # plt.xlim(1e0, 1e3)
        plt.ylim(1e1, 1e2)
        plt.savefig('Plots/phase_2_task_4_ii.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.errorbar(degrees[100]/k_max[0], dists[100]/deg_dist_theoretical_ra(degrees[100], m=4),  yerr=errors[100]/deg_dist_theoretical_pa(degrees[100], m=4), marker = '.', ls = ' ', color='b', label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000]/k_max[1], dists[1000]/deg_dist_theoretical_ra(degrees[1000], m=4),  yerr=errors[1000]/deg_dist_theoretical_pa(degrees[1000], m=4), marker = '.', ls = ' ', color='g', label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000]/k_max[2], dists[10000]/deg_dist_theoretical_ra(degrees[10000], m=4),  yerr=errors[10000]/deg_dist_theoretical_pa(degrees[10000], m=4), marker = '.', ls = ' ', color='r', label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000]/k_max[3], dists[100000]/deg_dist_theoretical_ra(degrees[100000], m=4), yerr=errors[100000]/deg_dist_theoretical_pa(degrees[100000], m=4), marker = '.', ls = ' ', color='c', label=r'$Data: \: N=100000$')        
        plt.legend()
        plt.xlabel(r'$\it{k/k_1}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k) \: / \: p_{\infty}(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/phase_2_task_4_iii.png')

        plt.show()


def phase_3_task_1(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e4)
    if compute:
        for m in ms:
            graph, options = initialise_graph(size=m+1, m=m)
            
            bar = Bar('Code Running', max=size)
            for i in range(size):
                graph, options = add_vertex(graph, options, m=m, method='mi', q=2/3)
                bar.next()
            bar.finish()
            
            degrees = update_degrees(graph)

            degree_dist = Counter(list(degrees.values()))
            degrees = list(degree_dist.keys())
            occurence = list(degree_dist.values())
            
            file = open('Files/Phase3/phase_3_task_1_m'+str(m)+'.txt', 'wb')
            pickle.dump(degrees, file)
            pickle.dump(occurence, file)
            file.close()
    
    if plot:
        degrees = {}
        occurence = {}
        for m in ms:
            file = open('Files/Phase3/phase_3_task_1_m'+str(m)+'.txt', 'rb')
            degrees[m] = pickle.load(file)
            occurence[m] = pickle.load(file)
            file.close()

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(degrees[2], np.array(occurence[2])/size, '.', label=r'$m=2$', color='k')
        plt.plot(degrees[4], np.array(occurence[4])/size, '.', label=r'$m=4$', color='b')
        plt.plot(degrees[8], np.array(occurence[8])/size, '.', label=r'$m=8$', color='r')
        plt.plot(degrees[16], np.array(occurence[16])/size, '.', label=r'$m=16$', color='g')
        plt.plot(degrees[32], np.array(occurence[32])/size, '.', label=r'$m=32$', color='c')
        plt.plot(degrees[64], np.array(occurence[64])/size, '.', label=r'$m=64$', color='m')
        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_mi_2_3(x_space[0], m=2), '--', color='k')
        plt.plot(x_space[1], deg_dist_theoretical_mi_2_3(x_space[1], m=4), '--', color='b')
        plt.plot(x_space[2], deg_dist_theoretical_mi_2_3(x_space[2], m=8), '--', color='r')
        plt.plot(x_space[3], deg_dist_theoretical_mi_2_3(x_space[3], m=16), '--', color='g')
        plt.plot(x_space[4], deg_dist_theoretical_mi_2_3(x_space[4], m=32), '--', color='c')
        plt.plot(x_space[5], deg_dist_theoretical_mi_2_3(x_space[5], m=64), '--', color='m')
        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        # plt.xlim(1e0, 2e3)
        # plt.ylim(1e-4, 1e0)
        plt.savefig('Plots/phase_3_task_1_i.png')
        plt.show()


def phase_3_task_3(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e4)
    repetitions = [10*2**(6-n) for n in range(0, 6)]

    if compute:
        bar = Bar('Code Running', max=int(size * sum(repetitions)))

        for i, m in enumerate(ms):
            master_array = []
            big_x = []
            big_y = []

            for _ in range(repetitions[i]):
                graph, options = initialise_graph(size=(m+1), m=m)

                for _ in range(size):
                    graph, options = add_vertex(graph, options, m=m, method='mi', q=2/3)
                    bar.next()

                degrees = update_degrees(graph)
                master_array.append(list(degrees.values()))
                x, y = logbin(list(degrees.values()), scale=1.2)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.2)
            errors = combine_log_bins(big_x, big_y)

            file = open('Files/Phase3/phase_3_task_3_m'+str(m)+'.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        bar.finish()

    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for m in ms:
            file = open('Files/Phase3/phase_3_task_3_m'+str(m)+'.txt', 'rb')
            degrees[m] = pickle.load(file)
            dists[m] = pickle.load(file)
            errors[m] = pickle.load(file)
            file.close()

        r_sq = [r2_score(deg_dist_theoretical_mi_2_3(degrees[m], m=m), dists[m]) for m in ms]
        
        chi_sq = []
        for m in ms:
            observed = size*np.array(dists[m])
            indices = np.argwhere(observed<5)
            indices = np.ndarray.flatten(indices)
            observed = observed[observed>=5]
            theoretical = np.delete(size*deg_dist_theoretical_mi_2_3(degrees[m], m=m), indices)
            chi_sq.append(st.chisquare(observed, theoretical))
        
        ks_values = [st.ks_2samp(deg_dist_theoretical_mi_2_3(degrees[m], m=m), dists[m]) for m in ms]

        print('R2 values: ', r_sq)
        print('Chi_2 values: ', chi_sq)
        print('KS Test values: ', ks_values)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'  

        plt.errorbar(degrees[2], dists[2], yerr=errors[2], marker = '.', ls = ' ', color='k', label=r'$m=2$')
        plt.errorbar(degrees[4], dists[4], yerr=errors[4], marker = '.', ls = ' ', color='b', label=r'$m=4$')
        plt.errorbar(degrees[8], dists[8], yerr=errors[8], marker = '.', ls = ' ', color='r', label=r'$m=8$')
        plt.errorbar(degrees[16], dists[16], yerr=errors[16], marker = '.', ls = ' ', color='g', label=r'$m=16$')
        plt.errorbar(degrees[32], dists[32], yerr=errors[32], marker = '.', ls = ' ', color='c', label=r'$m=32$')
        plt.errorbar(degrees[64], dists[64], yerr=errors[64], marker = '.', ls = ' ', color='m', label=r'$m=64$')

        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_mi_2_3(x_space[0], m=2), '--', color='k')
        plt.plot(x_space[1], deg_dist_theoretical_mi_2_3(x_space[1], m=4), '--', color='b')
        plt.plot(x_space[2], deg_dist_theoretical_mi_2_3(x_space[2], m=8), '--', color='r')
        plt.plot(x_space[3], deg_dist_theoretical_mi_2_3(x_space[3], m=16), '--', color='g')
        plt.plot(x_space[4], deg_dist_theoretical_mi_2_3(x_space[4], m=32), '--', color='c')
        plt.plot(x_space[5], deg_dist_theoretical_mi_2_3(x_space[5], m=64), '--', color='m')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        # plt.xlim(1e0, 1e3)
        # plt.ylim(1e-2, 1e5)
        plt.savefig('Plots/phase_3_task_3_i.png')

        plt.show()


def phase_3_task_4(compute=True, plot=False):
    m = 4
    N = [10**n for n in range(2, 6)]
    repetitions = [10**(5-n) for n in range(1, 5)]

    if compute:
        bar = Bar('Code Running', max=int(np.dot(np.array(N), np.array(repetitions))))

        k_max = []
        k_err = []
        for i, n in enumerate(N):
            master_array = []
            big_x = []
            big_y = []
            
            k_s = []
            for r in range(repetitions[i]):
                graph, options = initialise_graph(size=(m+1), m=m)

                for _ in range(n):
                    graph, options = add_vertex(graph, options, m=m, method='mi', q=2/3)    
                    bar.next()

                degrees = update_degrees(graph)
                
                k_s.append(max(list(degrees.values())))
                
                master_array.append(list(degrees.values()))
                x, y = logbin(list(degrees.values()), scale=1.2)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.2)
            errors = combine_log_bins(big_x, big_y)
            k_max.append(np.average(k_s))
            k_err.append(np.std(k_s)/np.sqrt(len(k_s)))

            file = open('Files/Phase3/phase_3_task_4_m3_N'+str(n)+'.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        file = open('Files/Phase3/phase_3_task_4_k.txt', 'wb')
        pickle.dump(k_max, file)
        pickle.dump(k_err, file)
        file.close()


    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for n in N:
            file = open('Files/Phase3/phase_3_task_4_m3_N'+str(n)+'.txt', 'rb')
            degrees[n] = pickle.load(file)
            dists[n] = pickle.load(file)
            errors[n] = pickle.load(file)
            file.close()
        file = open('Files/Phase3/phase_3_task_4_k.txt', 'rb')
        k_max = pickle.load(file)
        k_err = pickle.load(file)
        
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'       
        plt.errorbar(degrees[100], dists[100],  yerr=errors[100], marker = '.', ls = ' ', color='b', label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000], dists[1000],  yerr=errors[1000], marker = '.', ls = ' ', color='g', label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000], dists[10000],  yerr=errors[10000], marker = '.', ls = ' ', color='r', label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000], dists[100000], yerr=errors[100000], marker = '.', ls = ' ', color='c', label=r'$Data: \: N=100000$')

        x_space = np.linspace(min(degrees[100]), max(degrees[100000]), 1000)
        plt.plot(x_space, deg_dist_theoretical_mi_2_3(x_space, m=4), '--', color='k', label=r'$Theoretical \: Data: \: m=4$')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{n(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/phase_3_task_4_i.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.errorbar(N, k_max,  yerr=k_err, marker = '.', ls = ' ', color='b', label=r'$Data$')

        plt.legend()
        plt.xlabel(r'$\it{N}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{k_1}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_off()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        # plt.xlim(1e0, 1e3)
        plt.ylim(1e1, 1e3)
        plt.savefig('Plots/phase_3_task_4_ii.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.errorbar(degrees[100]/k_max[0], dists[100]/deg_dist_theoretical_mi_2_3(degrees[100], m=4),  yerr=errors[100]/deg_dist_theoretical_pa(degrees[100], m=4), marker = '.', ls = ' ', color='b', label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000]/k_max[1], dists[1000]/deg_dist_theoretical_mi_2_3(degrees[1000], m=4),  yerr=errors[1000]/deg_dist_theoretical_pa(degrees[1000], m=4), marker = '.', ls = ' ', color='g', label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000]/k_max[2], dists[10000]/deg_dist_theoretical_mi_2_3(degrees[10000], m=4),  yerr=errors[10000]/deg_dist_theoretical_pa(degrees[10000], m=4), marker = '.', ls = ' ', color='r', label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000]/k_max[3], dists[100000]/deg_dist_theoretical_mi_2_3(degrees[100000], m=4), yerr=errors[100000]/deg_dist_theoretical_pa(degrees[100000], m=4), marker = '.', ls = ' ', color='c', label=r'$Data: \: N=100000$')        
        plt.legend()
        plt.xlabel(r'$\it{k/k_1}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k) \: / \: p_{\infty}(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/phase_3_task_4_iii.png')

        plt.show()


def combine_log_bins(data_x, data_y):
    data_x_final = np.unique(np.concatenate(data_x, 0))
    data_y_final = []
    errors = []

    for i in range(len(data_x_final)):
        sample = []
        for j in range(len(data_y)):
            if data_x_final[i] in data_x[j]:
                sample.append(data_y[j][data_x[j].index(data_x_final[i])])
        
        data_y_final.append(np.average(sample))
        errors.append(np.std(sample)/np.sqrt(len(sample)))

    return errors


def loading():
    # Loading symbol while running code for terminal use
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
        tasks = [1, 3, 4]
        if args.task not in tasks:  # If task No. provided is not valid raise value error
            raise ValueError("Phase No. provided not valid, must be either 1 or 2")
        else:
            if args.task == 1 and args.execute:
                phase_1_task_1(compute=True, plot=False)
            elif args.task == 1 and args.execute == False:
                phase_1_task_1(compute=False, plot=True)
            elif args.task == 3 and args.execute:
                phase_1_task_3(compute=True, plot=False)
            elif args.task == 3 and args.execute == False:
                phase_1_task_3(compute=False, plot=True)
            elif args.task == 4 and args.execute:
                phase_1_task_4(compute=True, plot=False)
            elif args.task == 4 and args.execute == False:
                phase_1_task_4(compute=False, plot=True)
            else:
                raise ValueError("Does not exist. Please enter -t 1, 3 or 4")
    elif args.phase_number == 2:
        tasks = [1, 3, 4]
        if args.task not in tasks:  # If task No. provided is not valid raise value error
            raise ValueError("Phase No. provided not valid, must be either 1 or 2")
        else:
            if args.task == 1 and args.execute:
                phase_2_task_1(compute=True, plot=False)
            elif args.task == 1 and args.execute == False:
                phase_2_task_1(compute=False, plot=True)
            elif args.task == 3 and args.execute:
                phase_2_task_3(compute=True, plot=False)
            elif args.task == 3 and args.execute == False:
                phase_2_task_3(compute=False, plot=True)
            elif args.task == 4 and args.execute:
                phase_2_task_4(compute=True, plot=False)
            elif args.task == 4 and args.execute == False:
                phase_2_task_4(compute=False, plot=True)
            else:
                raise ValueError("Does not exist. Please enter -t 1, 3 or 4")
    elif args.phase_number == 3:
        tasks = [1, 3, 4]
        if args.task not in tasks:  # If task No. provided is not valid raise value error
            raise ValueError("Phase No. provided not valid, must be either 1 or 2")
        else:
            if args.task == 1 and args.execute:
                phase_3_task_1(compute=True, plot=False)
            elif args.task == 1 and args.execute == False:
                phase_3_task_1(compute=False, plot=True)
            elif args.task == 3 and args.execute:
                phase_3_task_3(compute=True, plot=False)
            elif args.task == 3 and args.execute == False:
                phase_3_task_3(compute=False, plot=True)
            elif args.task == 4 and args.execute:
                phase_3_task_4(compute=True, plot=False)
            elif args.task == 4 and args.execute == False:
                phase_3_task_4(compute=False, plot=True)
            else:
                raise ValueError("Does not exist. Please enter -t 1, 3 or 4")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Complexity & Networks: Networks Project - Main Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-p', '--phase_number', type=int, help='Specify Phase Number')
    parser.add_argument('-t', '--task', type=int, help='Task number to be executed')
    parser.add_argument('-e', '--execute', action='store_true', help='Flag: if present will execute rather than plot task')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided