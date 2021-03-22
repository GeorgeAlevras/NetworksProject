import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

n = np.linspace(0, 100, 100)

f = lambda x: x
g = lambda x: 2*x

y1 = f(n)
y2 = g(n)

fig, ax = plt.subplots()
p1, = plt.plot(n, y1, '--')
p2, = plt.plot(n, y2, '.')
l = plt.legend([(p1, p2)], ['Two keys'], numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})
plt.show()
