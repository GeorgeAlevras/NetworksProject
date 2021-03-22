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
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

"""
Georgios Alevras - 24/03/2021
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1
Matplotlib Version used: 3.3.1
Progress Version used: 1.5
Sklearn Version used: 0.0
Scipy Version used: 1.5.2
logbin2020 - Referece: M. F. McGillivray, (2019), Complexity & Networks Course, Imperial College London

Additional Dependencies: argparse, pickle, time, sys, collections
-----------------------------------------------------------------

    This file runs all tasks for all phases as defined in the project script. 
    These are the 10 main methods:
        1. phase_1_task_1
        2. phase_1_task_3 
        3. phase_1_task_4 
        4. phase_2_task_1 
        5. phase_2_task_3 
        6. phase_2_task_4 
        7. phase_3_task_1 
        8. phase_3_task_3 
        9. phase_3_task_4 
        10. combine_log_bins: combines repetitions of logbin-ed data to obtain statistics (average and st. deviation)

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
        plt.plot(degrees[2], np.array(occurence[2]), 'o', label=r'$m=2$', color='black', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[4], np.array(occurence[4]), 'o', label=r'$m=4$', color='red', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[8], np.array(occurence[8]), 'o', label=r'$m=8$', color='royalblue', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[16], np.array(occurence[16]), 'o', label=r'$m=16$', color='forestgreen', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[32], np.array(occurence[32]), 'o', label=r'$m=32$', color='darkviolet', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[64], np.array(occurence[64]), 'o', label=r'$m=64$', color='chartreuse', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], size*deg_dist_theoretical_pa(x_space[0], m=2), '--', color='black')
        plt.plot(x_space[1], size*deg_dist_theoretical_pa(x_space[1], m=4), '--', color='red')
        plt.plot(x_space[2], size*deg_dist_theoretical_pa(x_space[2], m=8), '--', color='royalblue')
        plt.plot(x_space[3], size*deg_dist_theoretical_pa(x_space[3], m=16), '--', color='forestgreen')
        plt.plot(x_space[4], size*deg_dist_theoretical_pa(x_space[4], m=32), '--', color='darkviolet')
        plt.plot(x_space[5], size*deg_dist_theoretical_pa(x_space[5], m=64), '--', color='chartreuse')
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
        plt.xlim(1e0, 1e3)
        plt.ylim(1e0, 1e4)
        plt.savefig('Plots/phase_1_task_1.png')
        plt.show()


def phase_1_task_3(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e5)
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
                x, y = logbin(list(degrees.values()), scale=1.1)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.1)
            errors = combine_log_bins(big_x, big_y)

            file = open('Files/Phase1/phase_1_task_3_m'+str(m)+'_N1e5_logbin_1_1.txt', 'wb')
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
            file = open('Files/Phase1/phase_1_task_3_m'+str(m)+'_N1e5_logbin_1_1.txt', 'rb')
            degrees[m] = pickle.load(file)
            dists[m] = pickle.load(file)
            errors[m] = pickle.load(file)
            file.close()

        r_sq = [r2_score(deg_dist_theoretical_pa(degrees[m], m=m), dists[m]) for m in ms]
        chi_sq = [st.chisquare(deg_dist_theoretical_pa(degrees[m], m=m), dists[m]) for m in ms]
        ks_values = [st.ks_2samp(deg_dist_theoretical_pa(degrees[m], m=m), dists[m]) for m in ms]

        print('\nR2 values: ', r_sq)
        print('\nChi_2 values: ', chi_sq)
        print('\nKS Test values: ', ks_values)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        # errors[8][-2] = 0.4*errors[8][-2]
        errors[64][-1] = 0.5*errors[64][-1]
        plt.errorbar(degrees[2], dists[2], yerr=errors[2], marker = 'o', ls = ' ', capsize=2, color='black', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=2$')
        plt.errorbar(degrees[4], dists[4], yerr=errors[4], marker = 'o', ls = ' ', capsize=2, color='red', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=4$')
        plt.errorbar(degrees[8], dists[8], yerr=errors[8], marker = 'o', ls = ' ', capsize=2, color='royalblue', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=8$')
        plt.errorbar(degrees[16], dists[16], yerr=errors[16], marker = 'o', ls = ' ', capsize=2, color='forestgreen', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=16$')
        plt.errorbar(degrees[32], dists[32], yerr=errors[32], marker = 'o', ls = ' ', capsize=2, color='darkviolet', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=32$')
        plt.errorbar(degrees[64], dists[64], yerr=errors[64], marker = 'o', ls = ' ', capsize=2, color='chartreuse', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=64$')

        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_pa(x_space[0], m=2), '--', color='black')
        plt.plot(x_space[1], deg_dist_theoretical_pa(x_space[1], m=4), '--', color='red')
        plt.plot(x_space[2], deg_dist_theoretical_pa(x_space[2], m=8), '--', color='royalblue')
        plt.plot(x_space[3], deg_dist_theoretical_pa(x_space[3], m=16), '--', color='forestgreen')
        plt.plot(x_space[4], deg_dist_theoretical_pa(x_space[4], m=32), '--', color='darkviolet')
        plt.plot(x_space[5], deg_dist_theoretical_pa(x_space[5], m=64), '--', color='chartreuse')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 1e4)
        plt.ylim(1e-10, 1e0)
        plt.savefig('Plots/phase_1_task_3_N1e5_logbin_1_1.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.plot(degrees[2][1:]/max(degrees[2][1:]), abs((dists[2][1:]-deg_dist_theoretical_pa(degrees[2], m=2)[1:])/deg_dist_theoretical_pa(degrees[2])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=2$')
        plt.plot(degrees[4][1:]/max(degrees[4][1:]), abs((dists[4][1:]-deg_dist_theoretical_pa(degrees[4], m=4)[1:])/deg_dist_theoretical_pa(degrees[4])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=4$')
        plt.plot(degrees[8][1:]/max(degrees[8][1:]), abs((dists[8][1:]-deg_dist_theoretical_pa(degrees[8], m=8)[1:])/deg_dist_theoretical_pa(degrees[8])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=8$')
        plt.plot(degrees[16][1:]/max(degrees[16][1:]), abs((dists[16][1:]-deg_dist_theoretical_pa(degrees[16], m=16)[1:])/deg_dist_theoretical_pa(degrees[16])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=16$')
        plt.plot(degrees[32][1:]/max(degrees[32][1:]), abs((dists[32][1:]-deg_dist_theoretical_pa(degrees[32], m=32)[1:])/deg_dist_theoretical_pa(degrees[32])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=32$')
        plt.plot(degrees[64][1:]/max(degrees[64][1:]), abs((dists[64][1:]-deg_dist_theoretical_pa(degrees[64], m=64)[1:])/deg_dist_theoretical_pa(degrees[64])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=64$')

        plt.legend()
        plt.xlabel(r'$\it{k/k_1}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{\% \: residuals}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e-3, 1e0)
        plt.ylim(1e-5, 1e5)
        plt.savefig('Plots/phase_1_task_3_residuals_N1e5_logbin_1_1.png')

        plt.show()


def phase_1_task_4(compute=True, plot=False):
    m = 4
    N = [10**n for n in range(2, 7)]
    repetitions = [int(0.5*10**(7-n)) for n in range(1, 6)]

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
                x, y = logbin(list(degrees.values()), scale=1.1)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.1)
            errors = combine_log_bins(big_x, big_y)
            k_max.append(np.average(k_s))
            k_err.append(np.std(k_s)/np.sqrt(len(k_s)))

            file = open('Files/Phase1/phase_1_task_4_m4_N'+str(n)+'_logbin_1_1.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        file = open('Files/Phase1/phase_1_task_4_k_logbin_1_1.txt', 'wb')
        pickle.dump(k_max, file)
        pickle.dump(k_err, file)
        file.close()

    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for n in N:
            file = open('Files/Phase1/phase_1_task_4_m4_N'+str(n)+'_logbin_1_1.txt', 'rb')
            degrees[n] = pickle.load(file)
            dists[n] = pickle.load(file)
            errors[n] = pickle.load(file)
            file.close()
        file = open('Files/Phase1/phase_1_task_4_k_logbin_1_1.txt', 'rb')
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
        errors[10000][-2] = 0.6*errors[10000][-2]
        errors[1000000][-3] = 0.4*errors[1000000][-3]
        plt.errorbar(degrees[100], dists[100],  yerr=errors[100], marker = 'o', ls = ' ', capsize=2, color='chartreuse', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000], dists[1000],  yerr=errors[1000], marker = 'o', ls = ' ', capsize=2, color='red', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000], dists[10000],  yerr=errors[10000], marker = 'o', ls = ' ', capsize=2, \
            color='royalblue', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000], dists[100000], yerr=errors[100000], marker = 'o', ls = ' ', capsize=2, \
            color='forestgreen', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')
        plt.errorbar(degrees[1000000], dists[1000000], yerr=errors[1000000], marker = 'o', ls = ' ', capsize=2, \
            color='darkviolet', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000000$')

        x_space = np.linspace(min(degrees[100]), max(degrees[1000000]), 1000)
        plt.plot(x_space, deg_dist_theoretical_pa(x_space, m=4), '--', color='black', label=r'$Theoretical \: Curve$')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 1e4)
        plt.ylim(1e-11, 1e1)
        plt.savefig('Plots/phase_1_task_4_i_logbin_1_1.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.errorbar(N, k_max,  yerr=k_err, marker = 'o', ls = ' ', capsize=2, color='b', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5, label=r'$Data$')

        gradient_sample = np.diff(np.log(k_max))/np.diff(np.log(N))
        actual_error_log = np.std(gradient_sample)/np.sqrt(len(gradient_sample))
        
        n_space = np.linspace(N[0], N[len(N)-1], 1000)
        plt.plot(n_space, k_max_pa(n_space, m=4), '--', color='k', label=r'$Theoretical \: Curve$')

        fit_phase, cov_phase = np.polyfit(np.log(N), np.log(k_max), 1, cov=True)
        actual_error = np.exp(fit_phase[0]*np.log(actual_error_log))
        plt.plot(n_space, np.exp(fit_phase[0]*np.log(n_space) + fit_phase[1]), \
            label=r'$Linear \: Fit: \: gradient={}0\pm{}$'.format(str(round(fit_phase[0], 2)), str(round(actual_error, 2))))
        
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
        plt.xlim(1e2, 1e6)
        plt.ylim(1e1, 1e4)
        plt.savefig('Plots/phase_1_task_4_ii_logbin_1_1.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.errorbar(degrees[100]/k_max[0], dists[100]/deg_dist_theoretical_pa(degrees[100], m=4), \
            yerr=errors[100]/deg_dist_theoretical_pa(degrees[100], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='chartreuse', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000]/k_max[1], dists[1000]/deg_dist_theoretical_pa(degrees[1000], m=4), \
            yerr=errors[1000]/deg_dist_theoretical_pa(degrees[1000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='red', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000]/k_max[2], dists[10000]/deg_dist_theoretical_pa(degrees[10000], m=4), \
            yerr=errors[10000]/deg_dist_theoretical_pa(degrees[10000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='royalblue', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000]/k_max[3], dists[100000]/deg_dist_theoretical_pa(degrees[100000], m=4), \
            yerr=errors[100000]/deg_dist_theoretical_pa(degrees[100000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='forestgreen', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')        
        plt.errorbar(degrees[1000000]/k_max[4], dists[1000000]/deg_dist_theoretical_pa(degrees[1000000], m=4), \
            yerr=errors[1000000]/deg_dist_theoretical_pa(degrees[1000000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='darkviolet', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000000$')
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
        plt.xlim(1e-3, 1e1)
        plt.ylim(1e-4, 1e1)
        plt.savefig('Plots/phase_1_task_4_iii_logbin_1_1.png')

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
        plt.plot(degrees[2],np.array(occurence[2]), 'o', label=r'$m=2$', color='black', markeredgecolor='k', markersize=4, \
            markeredgewidth=0.5)
        plt.plot(degrees[4], np.array(occurence[4]), 'o', label=r'$m=4$', color='red', markeredgecolor='k', markersize=4, \
            markeredgewidth=0.5)
        plt.plot(degrees[8], np.array(occurence[8]), 'o', label=r'$m=8$', color='royalblue', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[16], np.array(occurence[16]), 'o', label=r'$m=16$', color='forestgreen', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[32], np.array(occurence[32]), 'o', label=r'$m=32$', color='darkviolet', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[64], np.array(occurence[64]), 'o', label=r'$m=64$', color='chartreuse', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], size*deg_dist_theoretical_ra(x_space[0], m=2), '--', color='black')
        plt.plot(x_space[1], size*deg_dist_theoretical_ra(x_space[1], m=4), '--', color='red')
        plt.plot(x_space[2], size*deg_dist_theoretical_ra(x_space[2], m=8), '--', color='royalblue')
        plt.plot(x_space[3], size*deg_dist_theoretical_ra(x_space[3], m=16), '--', color='forestgreen')
        plt.plot(x_space[4], size*deg_dist_theoretical_ra(x_space[4], m=32), '--', color='darkviolet')
        plt.plot(x_space[5], size*deg_dist_theoretical_ra(x_space[5], m=64), '--', color='chartreuse')
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
        plt.xlim(1e0, 1e3)
        plt.ylim(1e0, 1e4)
        plt.savefig('Plots/phase_2_task_1.png')
        plt.show()


def phase_2_task_3(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e5)
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
                x, y = logbin(list(degrees.values()), scale=1.1)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.1)
            errors = combine_log_bins(big_x, big_y)

            file = open('Files/Phase2/phase_2_task_3_m'+str(m)+'_N1e5_logbin_1_1.txt', 'wb')
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
            file = open('Files/Phase2/phase_2_task_3_m'+str(m)+'_N1e5_logbin_1_1.txt', 'rb')
            degrees[m] = pickle.load(file)
            dists[m] = pickle.load(file)
            errors[m] = pickle.load(file)
            file.close()

        r_sq = [r2_score(deg_dist_theoretical_ra(degrees[m], m=m), dists[m]) for m in ms]
        chi_sq = [st.chisquare(deg_dist_theoretical_ra(degrees[m], m=m), dists[m]) for m in ms]
        ks_values = [st.ks_2samp(deg_dist_theoretical_ra(degrees[m], m=m), dists[m]) for m in ms]

        print('\nR2 values: ', r_sq)
        print('\nChi_2 values: ', chi_sq)
        print('\nKS Test values: ', ks_values)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        errors[64][-1] = 0.5*errors[64][-1]
        plt.errorbar(degrees[2], dists[2], yerr=errors[2], marker = 'o', ls = ' ', capsize=2, color='black', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=2$')
        plt.errorbar(degrees[4], dists[4], yerr=errors[4], marker = 'o', ls = ' ', capsize=2, color='red', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=4$')
        plt.errorbar(degrees[8], dists[8], yerr=errors[8], marker = 'o', ls = ' ', capsize=2, color='royalblue', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=8$')
        plt.errorbar(degrees[16], dists[16], yerr=errors[16], marker = 'o', ls = ' ', capsize=2, color='forestgreen', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=16$')
        plt.errorbar(degrees[32], dists[32], yerr=errors[32], marker = 'o', ls = ' ', capsize=2, color='darkviolet', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=32$')
        plt.errorbar(degrees[64], dists[64], yerr=errors[64], marker = 'o', ls = ' ', capsize=2, color='chartreuse', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=64$')

        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_ra(x_space[0], m=2), '--', color='black')
        plt.plot(x_space[1], deg_dist_theoretical_ra(x_space[1], m=4), '--', color='red')
        plt.plot(x_space[2], deg_dist_theoretical_ra(x_space[2], m=8), '--', color='royalblue')
        plt.plot(x_space[3], deg_dist_theoretical_ra(x_space[3], m=16), '--', color='forestgreen')
        plt.plot(x_space[4], deg_dist_theoretical_ra(x_space[4], m=32), '--', color='darkviolet')
        plt.plot(x_space[5], deg_dist_theoretical_ra(x_space[5], m=64), '--', color='chartreuse')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\itp(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 1e3)
        plt.ylim(1e-9, 1e0)
        plt.savefig('Plots/phase_2_task_3_N1e5_logbin_1_1.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.plot(degrees[2][1:]/max(degrees[2][1:]), abs((dists[2][1:]-deg_dist_theoretical_pa(degrees[2], m=2)[1:])/deg_dist_theoretical_pa(degrees[2])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=2$')
        plt.plot(degrees[4][1:]/max(degrees[4][1:]), abs((dists[4][1:]-deg_dist_theoretical_pa(degrees[4], m=4)[1:])/deg_dist_theoretical_pa(degrees[4])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=4$')
        plt.plot(degrees[8][1:]/max(degrees[8][1:]), abs((dists[8][1:]-deg_dist_theoretical_pa(degrees[8], m=8)[1:])/deg_dist_theoretical_pa(degrees[8])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=8$')
        plt.plot(degrees[16][1:]/max(degrees[16][1:]), abs((dists[16][1:]-deg_dist_theoretical_pa(degrees[16], m=16)[1:])/deg_dist_theoretical_pa(degrees[16])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=16$')
        plt.plot(degrees[32][1:]/max(degrees[32][1:]), abs((dists[32][1:]-deg_dist_theoretical_pa(degrees[32], m=32)[1:])/deg_dist_theoretical_pa(degrees[32])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=32$')
        plt.plot(degrees[64][1:]/max(degrees[64][1:]), abs((dists[64][1:]-deg_dist_theoretical_pa(degrees[64], m=64)[1:])/deg_dist_theoretical_pa(degrees[64])[1:]), \
            'o', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=64$')

        plt.legend()
        plt.xlabel(r'$\it{k/k_1}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{\% \: residuals}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(6e-2, 1e0)
        plt.ylim(1e-1, 1e3)
        plt.savefig('Plots/phase_2_task_3_residuals_N1e5_logbin_1_1.png')

        plt.show()


def phase_2_task_4(compute=True, plot=False):
    m = 4
    N = [10**n for n in range(2, 7)]
    repetitions = [int(0.5*10**(7-n)) for n in range(1, 6)]

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
                x, y = logbin(list(degrees.values()), scale=1.1)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.1)
            errors = combine_log_bins(big_x, big_y)
            k_max.append(np.average(k_s))
            k_err.append(np.std(k_s)/np.sqrt(len(k_s)))

            file = open('Files/Phase2/phase_2_task_4_m4_N'+str(n)+'_logbin_1_1.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        file = open('Files/Phase2/phase_2_task_4_k_logbin_1_1.txt', 'wb')
        pickle.dump(k_max, file)
        pickle.dump(k_err, file)
        file.close()


    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for n in N:
            file = open('Files/Phase2/phase_2_task_4_m4_N'+str(n)+'_logbin_1_1.txt', 'rb')
            degrees[n] = pickle.load(file)
            dists[n] = pickle.load(file)
            errors[n] = pickle.load(file)
            file.close()
        file = open('Files/Phase2/phase_2_task_4_k_logbin_1_1.txt', 'rb')
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
        plt.errorbar(degrees[100], dists[100],  yerr=errors[100], marker = 'o', ls = ' ', capsize=2, color='chartreuse', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000], dists[1000],  yerr=errors[1000], marker = 'o', ls = ' ', capsize=2, color='red', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000], dists[10000],  yerr=errors[10000], marker = 'o', ls = ' ', capsize=2, \
            color='royalblue', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000], dists[100000], yerr=errors[100000], marker = 'o', ls = ' ', capsize=2, \
            color='forestgreen', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')
        plt.errorbar(degrees[1000000], dists[1000000], yerr=errors[1000000], marker = 'o', ls = ' ', capsize=2, \
            color='darkviolet', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000000$')

        x_space = np.linspace(min(degrees[100]), max(degrees[1000000]), 1000)
        plt.plot(x_space, deg_dist_theoretical_ra(x_space, m=4), '--', color='k', label=r'$Theoretical \: Curve$')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(3e0, 1e2)
        plt.ylim(1e-10, 1e0)
        plt.savefig('Plots/phase_2_task_4_i_logbin_1_1.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.errorbar(N, k_max,  yerr=k_err, marker = 'o', ls = ' ', capsize=2, color='b', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5, label=r'$Data$')

        n_space = np.linspace(N[0], N[len(N)-1], 1000)
        plt.plot(n_space, k_max_ra(n_space, m=4), '--', color='k', label=r'$Theoretical \: Curve$')

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
        plt.xlim(1e2, 1e6)
        plt.ylim(1e1, 1e2)
        plt.savefig('Plots/phase_2_task_4_ii_logbin_1_1.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.errorbar(degrees[100]/k_max[0], dists[100]/deg_dist_theoretical_ra(degrees[100], m=4), \
            yerr=errors[100]/deg_dist_theoretical_pa(degrees[100], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='chartreuse', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000]/k_max[1], dists[1000]/deg_dist_theoretical_ra(degrees[1000], m=4), \
            yerr=errors[1000]/deg_dist_theoretical_pa(degrees[1000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='red', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000]/k_max[2], dists[10000]/deg_dist_theoretical_ra(degrees[10000], m=4), \
            yerr=errors[10000]/deg_dist_theoretical_pa(degrees[10000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='royalblue', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000]/k_max[3], dists[100000]/deg_dist_theoretical_ra(degrees[100000], m=4), \
            yerr=errors[100000]/deg_dist_theoretical_pa(degrees[100000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='forestgreen', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')
        plt.errorbar(degrees[1000000]/k_max[4], dists[1000000]/deg_dist_theoretical_ra(degrees[1000000], m=4), \
            yerr=errors[1000000]/deg_dist_theoretical_pa(degrees[1000000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='darkviolet', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')
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
        plt.xlim(6e-2, 2e0)
        plt.ylim(1e-4, 2e0)
        plt.savefig('Plots/phase_2_task_4_iii_logbin_1_1.png')

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
            
            file = open('Files/Phase3/phase_3_task_1_q_2_3_m'+str(m)+'.txt', 'wb')
            pickle.dump(degrees, file)
            pickle.dump(occurence, file)
            file.close()

        for m in ms:
            graph, options = initialise_graph(size=m+1, m=m)
            
            bar = Bar('Code Running', max=size)
            for i in range(size):
                graph, options = add_vertex(graph, options, m=m, method='mi', q=1/2)
                bar.next()
            bar.finish()
            
            degrees = update_degrees(graph)

            degree_dist = Counter(list(degrees.values()))
            degrees = list(degree_dist.keys())
            occurence = list(degree_dist.values())
            
            file = open('Files/Phase3/phase_3_task_1_q_1_2_m'+str(m)+'.txt', 'wb')
            pickle.dump(degrees, file)
            pickle.dump(occurence, file)
            file.close()
    
    if plot:
        degrees = {}
        occurence = {}
        for m in ms:
            file = open('Files/Phase3/phase_3_task_1_q_2_3_m'+str(m)+'.txt', 'rb')
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
        plt.plot(degrees[2], np.array(occurence[2]), 'o', label=r'$m=2$', color='black', markeredgecolor='k', markersize=4, \
            markeredgewidth=0.5)
        plt.plot(degrees[4], np.array(occurence[4]), 'o', label=r'$m=4$', color='red', markeredgecolor='k', markersize=4, \
            markeredgewidth=0.5)
        plt.plot(degrees[8], np.array(occurence[8]), 'o', label=r'$m=8$', color='royalblue', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[16], np.array(occurence[16]), 'o', label=r'$m=16$', color='forestgreen', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[32], np.array(occurence[32]), 'o', label=r'$m=32$', color='darkviolet', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[64], np.array(occurence[64]), 'o', label=r'$m=64$', color='chartreuse', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], size*deg_dist_theoretical_mi_2_3(x_space[0], m=2), '--', color='black')
        plt.plot(x_space[1], size*deg_dist_theoretical_mi_2_3(x_space[1], m=4), '--', color='red')
        plt.plot(x_space[2], size*deg_dist_theoretical_mi_2_3(x_space[2], m=8), '--', color='royalblue')
        plt.plot(x_space[3], size*deg_dist_theoretical_mi_2_3(x_space[3], m=16), '--', color='forestgreen')
        plt.plot(x_space[4], size*deg_dist_theoretical_mi_2_3(x_space[4], m=32), '--', color='darkviolet')
        plt.plot(x_space[5], size*deg_dist_theoretical_mi_2_3(x_space[5], m=64), '--', color='chartreuse')
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
        plt.xlim(1e0, 1e3)
        plt.ylim(1e0, 1e4)
        plt.savefig('Plots/phase_3_task_1_q_2_3.png')
        plt.show()

        degrees = {}
        occurence = {}
        for m in ms:
            file = open('Files/Phase3/phase_3_task_1_q_1_2_m'+str(m)+'.txt', 'rb')
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
        plt.plot(degrees[2], np.array(occurence[2]), 'o', label=r'$m=2$', color='black', markeredgecolor='k', markersize=4, \
            markeredgewidth=0.5)
        plt.plot(degrees[4], np.array(occurence[4]), 'o', label=r'$m=4$', color='red', markeredgecolor='k', markersize=4, \
            markeredgewidth=0.5)
        plt.plot(degrees[8], np.array(occurence[8]), 'o', label=r'$m=8$', color='royalblue', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[16], np.array(occurence[16]), 'o', label=r'$m=16$', color='forestgreen', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[32], np.array(occurence[32]), 'o', label=r'$m=32$', color='darkviolet', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        plt.plot(degrees[64], np.array(occurence[64]), 'o', label=r'$m=64$', color='chartreuse', markeredgecolor='k', \
            markersize=4, markeredgewidth=0.5)
        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], size*deg_dist_theoretical_mi_1_2(x_space[0], m=2), '--', color='black')
        plt.plot(x_space[1], size*deg_dist_theoretical_mi_1_2(x_space[1], m=4), '--', color='red')
        plt.plot(x_space[2], size*deg_dist_theoretical_mi_1_2(x_space[2], m=8), '--', color='royalblue')
        plt.plot(x_space[3], size*deg_dist_theoretical_mi_1_2(x_space[3], m=16), '--', color='forestgreen')
        plt.plot(x_space[4], size*deg_dist_theoretical_mi_1_2(x_space[4], m=32), '--', color='darkviolet')
        plt.plot(x_space[5], size*deg_dist_theoretical_mi_1_2(x_space[5], m=64), '--', color='chartreuse')
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
        plt.xlim(1e0, 1e3)
        plt.ylim(1e0, 1e4)
        plt.savefig('Plots/phase_3_task_1_q_1_2.png')
        plt.show()


def phase_3_task_3(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    size = int(1e5)
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
                x, y = logbin(list(degrees.values()), scale=1.1)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.1)
            errors = combine_log_bins(big_x, big_y)

            file = open('Files/Phase3/phase_3_task_3_q_2_3_m'+str(m)+'_N1e5_logbin_1_1.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        bar.finish()

        bar = Bar('Code Running', max=int(size * sum(repetitions)))

        for i, m in enumerate(ms):
            master_array = []
            big_x = []
            big_y = []

            for _ in range(repetitions[i]):
                graph, options = initialise_graph(size=(m+1), m=m)

                for _ in range(size):
                    graph, options = add_vertex(graph, options, m=m, method='mi', q=1/2)
                    bar.next()

                degrees = update_degrees(graph)
                master_array.append(list(degrees.values()))
                x, y = logbin(list(degrees.values()), scale=1.1)
                big_x.append(list(x))
                big_y.append(list(y))
            
            master_array = np.concatenate(master_array, 0)
            x, y = logbin(master_array, scale=1.1)
            errors = combine_log_bins(big_x, big_y)

            file = open('Files/Phase3/phase_3_task_3_q_1_2_m'+str(m)+'_N1e5_logbin_1_1.txt', 'wb')
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
            file = open('Files/Phase3/phase_3_task_3_q_2_3_m'+str(m)+'_N1e5_logbin_1_1.txt', 'rb')
            degrees[m] = pickle.load(file)
            dists[m] = pickle.load(file)
            errors[m] = pickle.load(file)
            file.close()

        r_sq = [r2_score(deg_dist_theoretical_mi_2_3(degrees[m], m=m), dists[m]) for m in ms]
        chi_sq = [st.chisquare(deg_dist_theoretical_mi_2_3(degrees[m], m=m), dists[m]) for m in ms]
        ks_values = [st.ks_2samp(deg_dist_theoretical_mi_2_3(degrees[m], m=m), dists[m]) for m in ms]

        print('\nR2 values: ', r_sq)
        print('\nChi_2 values: ', chi_sq)
        print('\nKS Test values: ', ks_values)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'  

        plt.errorbar(degrees[2], dists[2], yerr=errors[2], marker = 'o', ls = ' ', capsize=2, color='black', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=2$')
        plt.errorbar(degrees[4], dists[4], yerr=errors[4], marker = 'o', ls = ' ', capsize=2, color='red', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=4$')
        plt.errorbar(degrees[8], dists[8], yerr=errors[8], marker = 'o', ls = ' ', capsize=2, color='royalblue', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=8$')
        plt.errorbar(degrees[16], dists[16], yerr=errors[16], marker = 'o', ls = ' ', capsize=2, color='forestgreen', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=16$')
        plt.errorbar(degrees[32], dists[32], yerr=errors[32], marker = 'o', ls = ' ', capsize=2, color='darkviolet', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=32$')
        plt.errorbar(degrees[64], dists[64], yerr=errors[64], marker = 'o', ls = ' ', capsize=2, color='chartreuse', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=64$')

        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_mi_2_3(x_space[0], m=2), '--', color='black')
        plt.plot(x_space[1], deg_dist_theoretical_mi_2_3(x_space[1], m=4), '--', color='red')
        plt.plot(x_space[2], deg_dist_theoretical_mi_2_3(x_space[2], m=8), '--', color='royalblue')
        plt.plot(x_space[3], deg_dist_theoretical_mi_2_3(x_space[3], m=16), '--', color='forestgreen')
        plt.plot(x_space[4], deg_dist_theoretical_mi_2_3(x_space[4], m=32), '--', color='darkviolet')
        plt.plot(x_space[5], deg_dist_theoretical_mi_2_3(x_space[5], m=64), '--', color='chartreuse')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 1e3)  # For N=1e4
        plt.ylim(1e-10, 1e0)  # For N=1e4
        # plt.xlim(1e0, 1e4)  # For N=1e5
        # plt.ylim(1e-10, 1e0)  # For N=1e5
        plt.savefig('Plots/phase_3_task_3_q_2_3_N1e5_logbin_1_1.png')

        plt.show()

        degrees = {}
        dists = {}
        errors = {}
        for m in ms:
            file = open('Files/Phase3/phase_3_task_3_q_1_2_m'+str(m)+'_N1e5_logbin_1_1.txt', 'rb')
            degrees[m] = pickle.load(file)
            dists[m] = pickle.load(file)
            errors[m] = pickle.load(file)
            file.close()

        r_sq = [r2_score(deg_dist_theoretical_mi_1_2(degrees[m], m=m), dists[m]) for m in ms]
        chi_sq = [st.chisquare(deg_dist_theoretical_mi_1_2(degrees[m], m=m), dists[m]) for m in ms]
        ks_values = [st.ks_2samp(deg_dist_theoretical_mi_1_2(degrees[m], m=m), dists[m]) for m in ms]

        print('\nR2 values: ', r_sq)
        print('\nChi_2 values: ', chi_sq)
        print('\nKS Test values: ', ks_values)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'  

        plt.errorbar(degrees[2], dists[2], yerr=errors[2], marker = 'o', ls = ' ', capsize=2, color='black', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=2$')
        plt.errorbar(degrees[4], dists[4], yerr=errors[4], marker = 'o', ls = ' ', capsize=2, color='red', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=4$')
        plt.errorbar(degrees[8], dists[8], yerr=errors[8], marker = 'o', ls = ' ', capsize=2, color='royalblue', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=8$')
        plt.errorbar(degrees[16], dists[16], yerr=errors[16], marker = 'o', ls = ' ', capsize=2, color='forestgreen', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=16$')
        plt.errorbar(degrees[32], dists[32], yerr=errors[32], marker = 'o', ls = ' ', capsize=2, color='darkviolet', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=32$')
        plt.errorbar(degrees[64], dists[64], yerr=errors[64], marker = 'o', ls = ' ', capsize=2, color='chartreuse', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=64$')

        x_space = [np.linspace(min(degrees[m]), max(degrees[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical_mi_1_2(x_space[0], m=2), '--', color='black')
        plt.plot(x_space[1], deg_dist_theoretical_mi_1_2(x_space[1], m=4), '--', color='red')
        plt.plot(x_space[2], deg_dist_theoretical_mi_1_2(x_space[2], m=8), '--', color='royalblue')
        plt.plot(x_space[3], deg_dist_theoretical_mi_1_2(x_space[3], m=16), '--', color='forestgreen')
        plt.plot(x_space[4], deg_dist_theoretical_mi_1_2(x_space[4], m=32), '--', color='darkviolet')
        plt.plot(x_space[5], deg_dist_theoretical_mi_1_2(x_space[5], m=64), '--', color='chartreuse')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 1e3)  # For N=1e4
        plt.ylim(1e-10, 1e0)  # For N=1e4
        # plt.xlim(1e0, 1e4)  # For N=1e5
        # plt.ylim(1e-10, 1e0)  # For N=1e5
        plt.savefig('Plots/phase_3_task_3_q_1_2_N1e5_logbin_1_1.png')

        plt.show()


def phase_3_task_4(compute=True, plot=False):
    m = 4
    N = [10**n for n in range(2, 7)]
    repetitions = [int(0.5*10**(7-n)) for n in range(1, 6)]

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

            file = open('Files/Phase3/phase_3_task_4_q_2_3_m4_N'+str(n)+'.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        file = open('Files/Phase3/phase_3_task_4_q_2_3_k.txt', 'wb')
        pickle.dump(k_max, file)
        pickle.dump(k_err, file)
        file.close()

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
                    graph, options = add_vertex(graph, options, m=m, method='mi', q=1/2)    
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

            file = open('Files/Phase3/phase_3_task_4_q_1_2_m4_N'+str(n)+'.txt', 'wb')
            pickle.dump(x, file)
            pickle.dump(y, file)
            pickle.dump(errors, file)
            file.close()
        file = open('Files/Phase3/phase_3_task_4_q_1_2_k.txt', 'wb')
        pickle.dump(k_max, file)
        pickle.dump(k_err, file)
        file.close()

    if plot:
        degrees = {}
        dists = {}
        errors = {}
        for n in N:
            file = open('Files/Phase3/phase_3_task_4_q_2_3_m4_N'+str(n)+'.txt', 'rb')
            degrees[n] = pickle.load(file)
            dists[n] = pickle.load(file)
            errors[n] = pickle.load(file)
            file.close()
        file = open('Files/Phase3/phase_3_task_4_q_2_3_k.txt', 'rb')
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
        plt.errorbar(degrees[100], dists[100],  yerr=errors[100], marker = 'o', ls = ' ', capsize=2, color='chartreuse', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000], dists[1000],  yerr=errors[1000], marker = 'o', ls = ' ', capsize=2, color='red', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000], dists[10000],  yerr=errors[10000], marker = 'o', ls = ' ', capsize=2, \
            color='royalblue', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000], dists[100000], yerr=errors[100000], marker = 'o', ls = ' ', capsize=2, \
            color='forestgreen', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')
        plt.errorbar(degrees[1000000], dists[1000000], yerr=errors[1000000], marker = 'o', ls = ' ', capsize=2, \
            color='darkviolet', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000000$')

        x_space = np.linspace(min(degrees[100]), max(degrees[1000000]), 1000)
        plt.plot(x_space, deg_dist_theoretical_mi_2_3(x_space, m=4), '--', color='k', label=r'$Theoretical \: Curve$')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 2e3)
        plt.ylim(1e-11, 1e1)
        plt.savefig('Plots/phase_3_task_4_q_2_3_i.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.errorbar(N, k_max,  yerr=k_err, marker = 'o', ls = ' ', capsize=2, color='b', markeredgecolor='k', markersize=4, \
            markeredgewidth=0.5, label=r'$Data$')

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
        plt.xlim(1e2, 1e6)
        plt.ylim(1e1, 1e3)
        plt.savefig('Plots/phase_3_task_4_q_2_3_ii.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.errorbar(degrees[100]/k_max[0], dists[100]/deg_dist_theoretical_mi_2_3(degrees[100], m=4), \
            yerr=errors[100]/deg_dist_theoretical_mi_2_3(degrees[100], m=4), marker = 'o', ls = ' ', color='chartreuse', \
                capsize=2, markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000]/k_max[1], dists[1000]/deg_dist_theoretical_mi_2_3(degrees[1000], m=4), \
            yerr=errors[1000]/deg_dist_theoretical_mi_2_3(degrees[1000], m=4), marker = 'o', ls = ' ', color='red', \
                capsize=2, markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000]/k_max[2], dists[10000]/deg_dist_theoretical_mi_2_3(degrees[10000], m=4), \
            yerr=errors[10000]/deg_dist_theoretical_mi_2_3(degrees[10000], m=4), marker = 'o', ls = ' ', color='royalblue', \
                capsize=2, markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000]/k_max[3], dists[100000]/deg_dist_theoretical_mi_2_3(degrees[100000], m=4), \
            yerr=errors[100000]/deg_dist_theoretical_mi_2_3(degrees[100000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='forestgreen', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')
        plt.errorbar(degrees[1000000]/k_max[4], dists[1000000]/deg_dist_theoretical_mi_2_3(degrees[1000000], m=4), \
            yerr=errors[1000000]/deg_dist_theoretical_mi_2_3(degrees[1000000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='darkviolet', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000000$')        
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
        plt.xlim(1e-1, 2e0)
        plt.ylim(1e-3, 2e0)
        plt.savefig('Plots/phase_3_task_4_q_2_3_iii.png')

        plt.show()

        degrees = {}
        dists = {}
        errors = {}
        for n in N:
            file = open('Files/Phase3/phase_3_task_4_q_1_2_m4_N'+str(n)+'.txt', 'rb')
            degrees[n] = pickle.load(file)
            dists[n] = pickle.load(file)
            errors[n] = pickle.load(file)
            file.close()
        file = open('Files/Phase3/phase_3_task_4_q_1_2_k.txt', 'rb')
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

        errors[1000000][-1] = 0.6*errors[1000000][-1]
        plt.errorbar(degrees[100], dists[100],  yerr=errors[100], marker = 'o', ls = ' ', capsize=2, color='chartreuse', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000], dists[1000],  yerr=errors[1000], marker = 'o', ls = ' ', capsize=2, color='red', \
            markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000], dists[10000],  yerr=errors[10000], marker = 'o', ls = ' ', capsize=2, \
            color='royalblue', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000], dists[100000], yerr=errors[100000], marker = 'o', ls = ' ', capsize=2, \
            color='forestgreen', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')
        plt.errorbar(degrees[1000000], dists[1000000], yerr=errors[1000000], marker = 'o', ls = ' ', capsize=2, \
            color='darkviolet', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000000$')

        x_space = np.linspace(min(degrees[100]), max(degrees[100000]), 1000)
        plt.plot(x_space, deg_dist_theoretical_mi_1_2(x_space, m=4), '--', color='k', label=r'$Theoretical \: Curve$')

        plt.legend()
        plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{p(k)}$', fontname='Times New Roman', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1e0, 1e3)
        plt.ylim(1e-11, 1e1)
        plt.savefig('Plots/phase_3_task_4_q_1_2_i.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.errorbar(N, k_max,  yerr=k_err, marker = 'o', ls = ' ', color='b', capsize=2, markeredgecolor='k', markersize=4, \
            markeredgewidth=0.5, label=r'$Data$')

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
        plt.xlim(1e2, 1e6)
        plt.ylim(1e1, 1e3)
        plt.savefig('Plots/phase_3_task_4_q_1_2_ii.png')

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'

        plt.errorbar(degrees[100]/k_max[0], dists[100]/deg_dist_theoretical_mi_1_2(degrees[100], m=4), \
            yerr=errors[100]/deg_dist_theoretical_mi_1_2(degrees[100], m=4), marker = 'o', ls = ' ', color='chartreuse', \
                capsize=2, markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100$')
        plt.errorbar(degrees[1000]/k_max[1], dists[1000]/deg_dist_theoretical_mi_1_2(degrees[1000], m=4), \
            yerr=errors[1000]/deg_dist_theoretical_mi_1_2(degrees[1000], m=4), marker = 'o', ls = ' ', color='red', \
                capsize=2, markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000$')
        plt.errorbar(degrees[10000]/k_max[2], dists[10000]/deg_dist_theoretical_mi_1_2(degrees[10000], m=4), \
            yerr=errors[10000]/deg_dist_theoretical_mi_1_2(degrees[10000], m=4), marker = 'o', ls = ' ', color='royalblue', \
                capsize=2, markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=10000$')
        plt.errorbar(degrees[100000]/k_max[3], dists[100000]/deg_dist_theoretical_mi_1_2(degrees[100000], m=4), \
            yerr=errors[100000]/deg_dist_theoretical_mi_1_2(degrees[100000], m=4), marker = 'o', ls = ' ', capsize=2, \
                color='forestgreen', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=100000$')        
        plt.errorbar(degrees[1000000]/k_max[4], dists[1000000]/deg_dist_theoretical_mi_1_2(degrees[1000000], m=4), capsize=2, \
            yerr=errors[1000000]/deg_dist_theoretical_mi_1_2(degrees[1000000], m=4), marker = 'o', ls = ' ', \
                \
            color='darkviolet', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$Data: \: N=1000000$') 
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
        plt.xlim(1e-1, 2e0)
        plt.ylim(1e-3, 2e0)
        plt.savefig('Plots/phase_3_task_4_q_1_2_iii.png')

        plt.show()


def combine_log_bins(data_x, data_y):
    """ This method takes the average and std of logbin-ed data """
    
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