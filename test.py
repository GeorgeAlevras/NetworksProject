from model import initialise_graph, update_e, update_degrees, add_vertex, deg_dist_theoretical_pa, save_graph, \
    k_max_pa, deg_dist_theoretical_ra, k_max_ra, deg_dist_theoretical_mi_2_3, deg_dist_theoretical_mi_1_2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from logbin2020 import logbin
import matplotlib

# n = np.linspace(0, 100, 100)

# f = lambda x: x
# g = lambda x: 2*x

# y1 = f(n)
# y2 = g(n)

# fig, ax = plt.subplots()
# p1, = plt.plot(n, y1, '--')
# p2, = plt.plot(n, y2, '.')
# l = plt.legend([(p1, p2)], ['Two keys'], numpoints=1,
#                handler_map={tuple: HandlerTuple(ndivide=None)})
# plt.show()



m = 64
size = int(2e4)
repetitions = 100

master_array = []
master_array_m = []

for _ in range(repetitions):
    graph, options = initialise_graph(size=(m+1), m=m)
    graph_m, options_m = initialise_graph(size=(2*m+1), m=m)

    for _ in range(size):
        graph, options = add_vertex(graph, options, m=m)
        graph_m, options_m = add_vertex(graph_m, options_m, m=m)

    degrees = update_degrees(graph)
    master_array.append(list(degrees.values()))
    degrees_m = update_degrees(graph_m)
    master_array_m.append(list(degrees_m.values()))

master_array = np.concatenate(master_array, 0)
x, y = logbin(master_array, scale=1.1)
master_array_m = np.concatenate(master_array_m, 0)
x_m, y_m = logbin(master_array_m, scale=1.1)


fig, ax = plt.subplots()
params = {'legend.fontsize': 12}
plt.rcParams.update(params)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

plt.plot(x, y, 'o', color='red', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=32, \: init-size=m+1$')
plt.plot(x_m, y_m, 'o', color='chartreuse', markeredgecolor='k', markersize=4, markeredgewidth=0.5, label=r'$m=32, \: init-size=2m+1$')
x_space = np.linspace(min(x), max(x), 1000)
plt.plot(x_space, deg_dist_theoretical_pa(x_space, m=m), '--', color='black')

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
# plt.xlim(1e0, 1e4)
# plt.ylim(1e-10, 1e0)
plt.show()