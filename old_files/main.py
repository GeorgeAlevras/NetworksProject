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
        master_array = []
        
        m = 3
        size = int(1e4)
        for i in range(20):
            graph = initialise_graph(size=(m+1), m=m)
            degrees = update_degrees(graph)
            e = update_e(degrees)
            probabilities = update_probabilities_pa(graph, e)

            for i in range(size):
                graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=m, method='pa')    

            master_array.append(list(degrees.values()))
        
        master_array = np.concatenate(master_array, 0)
        x, y = logbin(master_array, scale=1.2)

        x_s = np.linspace(min(x), max(x), 1000)
        y_s = deg_dist_theoretical(x_s, m=3)

        fig, ax = plt.subplots()
        plt.plot(x, y, 'x')
        plt.plot(x_s, y_s, '--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.show()
        # degree_dist = Counter(list(degrees.values()))
        # degrees = list(degree_dist.keys())
        # occurence = list(degree_dist.values())
        # x_final, y_final, errors = combine_log_bins(master_data_x, master_data_y)


def phase_1_task_3(compute=True, plot=False):
    ms = [2, 4, 8, 16, 32, 64]
    
    if compute:
        size = int(1e4)
        repetitions = 50
        bar = Bar('Code Running', max=int(size * repetitions * len(ms)))
        
        for m in ms:
            master_data_x = []
            master_data_y = []
            
            for i in range(repetitions):
                graph = initialise_graph(size=(m+1), m=m)
                degrees = update_degrees(graph)
                e = update_e(degrees)
                probabilities = update_probabilities_pa(graph, e)

                for i in range(size):
                    graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=m, method='pa')    
                    bar.next()

                degrees_x, degrees_y = logbin(list(degrees.values()), scale=1.2)
                master_data_x.append(list(degrees_x))
                master_data_y.append(list(degrees_y))
                
                degree_dist = Counter(list(degrees.values()))
                degrees = list(degree_dist.keys())
                occurence = list(degree_dist.values())
            
            x_final, y_final, errors = combine_log_bins(master_data_x, master_data_y)

            file = open('Files/Phase1/phase_1_task_3_m'+str(m)+'.txt', 'wb')
            pickle.dump(degrees, file)
            pickle.dump(occurence, file)
            pickle.dump(x_final, file)
            pickle.dump(y_final, file)
            pickle.dump(errors, file)
            file.close()
        bar.finish()

    if plot:
        degrees_x = {}
        degrees_y = {}
        errors = {}
        
        for m in ms:
            file = open('Files/Phase1/phase_1_task_3_m'+str(m)+'.txt', 'rb')
            globals()['degrees_%s' %m] = pickle.load(file)
            globals()['occurence_%s' %m] = pickle.load(file)
            degrees_x[m] = pickle.load(file)
            degrees_y[m] = pickle.load(file)
            errors[m] = pickle.load(file)
            file.close()

        r_sq = [r2_score(deg_dist_theoretical(degrees_x[m], m=m), degrees_y[m]) for m in ms]
        chi_sq = []
        for m in ms:
            observed = 10000*np.array(degrees_y[m])
            indices = np.argwhere(observed<5)
            indices = np.ndarray.flatten(indices)
            observed = observed[observed>=5]
            theoretical = np.delete(10000*deg_dist_theoretical(degrees_x[m], m=m), indices)
            chi_sq.append(st.chisquare(observed, theoretical))
        
        print('R2 values: ', r_sq)
        print('Chi_2 values: ', chi_sq)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'       
        
        plt.errorbar(degrees_x[2], degrees_y[2], yerr=errors[2], marker = 'x', ls = ' ', color='k', label=r'$m=2$')
        plt.errorbar(degrees_x[4], degrees_y[4], yerr=errors[4], marker = 'x', ls = ' ', color='b', label=r'$m=4$')
        plt.errorbar(degrees_x[8], degrees_y[8], yerr=errors[8], marker = 'x', ls = ' ', color='r', label=r'$m=8$')
        plt.errorbar(degrees_x[16][1:], degrees_y[16][1:], yerr=errors[16][1:], marker = 'x', ls = ' ', color='g', label=r'$m=16$')
        plt.errorbar(degrees_x[32][1:], degrees_y[32][1:], yerr=errors[32][1:], marker = 'x', ls = ' ', color='c', label=r'$m=32$')
        plt.errorbar(degrees_x[64][1:], degrees_y[64][1:], yerr=errors[64][1:], marker = 'x', ls = ' ', color='m', label=r'$m=64$')
        
        x_space = [np.linspace(min(degrees_x[m]), max(degrees_x[m]), 1000) for m in ms]
        plt.plot(x_space[0], deg_dist_theoretical(x_space[0], m=2), '--', color='k')
        plt.plot(x_space[1], deg_dist_theoretical(x_space[1], m=4), '--', color='b')
        plt.plot(x_space[2], deg_dist_theoretical(x_space[2], m=8), '--', color='r')
        plt.plot(x_space[3], deg_dist_theoretical(x_space[3], m=16), '--', color='g')
        plt.plot(x_space[4], deg_dist_theoretical(x_space[4], m=32), '--', color='c')
        plt.plot(x_space[5], deg_dist_theoretical(x_space[5], m=64), '--', color='m')
        
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

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'      
        cum_m2 = np.cumsum(degrees_y[2])
        cum_m4 = np.cumsum(degrees_y[4])
        cum_m8 = np.cumsum(degrees_y[8])
        cum_m16 = np.cumsum(degrees_y[16])
        cum_m32 = np.cumsum(degrees_y[32])
        cum_m64 = np.cumsum(degrees_y[64])
        cum_exp_m2 = np.cumsum(deg_dist_theoretical(degrees_x[2], m=2))
        cum_exp_m4 = np.cumsum(deg_dist_theoretical(degrees_x[4], m=4))
        cum_exp_m8 = np.cumsum(deg_dist_theoretical(degrees_x[8], m=8))
        cum_exp_m16 = np.cumsum(deg_dist_theoretical(degrees_x[16], m=16))
        cum_exp_m32 = np.cumsum(deg_dist_theoretical(degrees_x[32], m=32))
        cum_exp_m64 = np.cumsum(deg_dist_theoretical(degrees_x[64], m=64))
        
        ks_m2 = st.ks_2samp(cum_exp_m2, cum_m2)
        ks_m4 = st.ks_2samp(cum_exp_m4, cum_m4)
        ks_m8 = st.ks_2samp(cum_exp_m8, cum_m8)
        ks_m16 = st.ks_2samp(cum_exp_m16, cum_m16)
        ks_m32 = st.ks_2samp(cum_exp_m32, cum_m32)
        ks_m64 = st.ks_2samp(cum_exp_m64, cum_m64)
        print('KS Test: ', ks_m2, ks_m4, ks_m8, ks_m16, ks_m32, ks_m64)
        
        plt.plot(degrees_x[2], cum_m2, 'o', label=r'$m=2 \: Data$')
        plt.plot(degrees_x[4], cum_m4, 'o', label=r'$m=4 \: Data$')
        plt.plot(degrees_x[8], cum_m8, 'o', label=r'$m=8 \: Data$')
        plt.plot(degrees_x[16], cum_m16, 'o', label=r'$m=16 \: Data$')
        plt.plot(degrees_x[32], cum_m32, 'o', label=r'$m=32 \: Data$')
        plt.plot(degrees_x[64], cum_m64, 'o', label=r'$m=64 \: Data$')
        plt.plot(degrees_x[2], cum_exp_m2, label=r'$m=2 \: Theory$')
        plt.plot(degrees_x[4], cum_exp_m4, label=r'$m=4 \: Theory$')
        plt.plot(degrees_x[8], cum_exp_m8, label=r'$m=8 \: Theory$')
        plt.plot(degrees_x[16], cum_exp_m16, label=r'$m=16 \: Theory$')
        plt.plot(degrees_x[32], cum_exp_m32, label=r'$m=32 \: Theory$')
        plt.plot(degrees_x[64], cum_exp_m64, label=r'$m=64 \: Theory$')
        
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
    m = 3
    N = [10**n for n in range(1, 4)]
    repetitions = [10**(4-n) for n in range(1, 4)]

    if compute:
        k_max = [0, 0, 0]
        for i, n in enumerate(N):
            master_data_x = []
            master_data_y = []
            
            for r in range(repetitions[i]):
                graph = initialise_graph(size=(m+1), m=m)
                degrees = update_degrees(graph)
                e = update_e(degrees)
                probabilities = update_probabilities_pa(graph, e)

                for _ in range(n):
                    graph, e, degrees, probabilities = add_vertex(graph, probabilities, m=m, method='pa')    

                if max(list(degrees.values())) > k_max[i]:
                    k_max[i] = max(list(degrees.values()))

                degrees_x, degrees_y = logbin(list(degrees.values()), scale=1.2)
                master_data_x.append(list(degrees_x))
                master_data_y.append(list(degrees_y))
                
                degree_dist = Counter(list(degrees.values()))
                degrees = list(degree_dist.keys())
                occurence = list(degree_dist.values())
            
            x_final, y_final, errors = combine_log_bins(master_data_x, master_data_y)

            file = open('Files/Phase1/phase_1_task_4_m3_N'+str(n)+'.txt', 'wb')
            pickle.dump(degrees, file)
            pickle.dump(occurence, file)
            pickle.dump(x_final, file)
            pickle.dump(y_final, file)
            pickle.dump(errors, file)
            file.close()
        print(k_max)

    if plot:
        degrees_x = {}
        degrees_y = {}
        errors = {}
        
        for n in N:
            file = open('Files/Phase1/phase_1_task_4_m3_N'+str(n)+'.txt', 'rb')
            globals()['degrees_%s' %n] = pickle.load(file)
            globals()['occurence_%s' %n] = pickle.load(file)
            degrees_x[n] = pickle.load(file)
            degrees_y[n] = pickle.load(file)
            errors[n] = pickle.load(file)
            file.close()
        
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'       
        plt.plot(degrees_x[10], degrees_y[10], 'x', label=r'$Data: \: N=10$')
        plt.plot(degrees_x[100], degrees_y[100], 'x', label=r'$Data: \: N=100$')
        plt.plot(degrees_x[1000], degrees_y[1000], 'x', label=r'$Data: \: N=1000$')

        x_space = np.linspace(min(degrees_x[10]), max(degrees_x[1000]), 1000)
        plt.plot(x_space, deg_dist_theoretical(x_space, m=3), '--', color='k', label=r'$Theoretical \: Data: \: m=3$')

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
        plt.savefig('Plots/phase_1_task_4.png')
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

    return data_x_final, data_y_final, errors


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