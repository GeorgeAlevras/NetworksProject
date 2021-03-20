import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def deg_dist_theoretical_ra(k, m=2):
    return ((m/(m+1))**(k-m))*(1/(1+m))

n = np.linspace(80, 500, 1000)

y = deg_dist_theoretical_ra(n, m=64)

fig, ax = plt.subplots()
plt.plot(n, y, '--')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
