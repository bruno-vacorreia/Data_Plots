"""
New Planet School

Let's learn how to add ERROR BARS to our data!
This Python script plots (fake) experimental data,
including both y error bars and a (fake) theoretical curve.
"""

# we always need these...
import numpy as np
import matplotlib.pyplot as plt

# data is entered into regular Python lists
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
y = [0.9, 0.8, 0.366, 0.5, 0.39, 0.45, 0.3, 0.15]

# including the error bar at each point
x_err = [0.1, 0.03, 0.05, 0.1, 0.03, 0.1, 0.01, 0.1]
y_err = [0.12, 0.05, 0.15, 0.17, 0.07, 0.2, 0.1, 0.14]

# add to plot the data as (x,y) with error bars
# note that xerr is also allowed; you can have both
plt.errorbar(x, y, yerr=y_err, xerr=x_err, lw=2, color='g', fmt='o--', capthick=2, label='data')

# let's also compare with the theoretical curve...
xx = np.linspace(0, 1, 50)
yy = np.exp(-3.*xx/2)
# ...and add that to the plot as well
plt.plot(xx, yy, lw=3, color='magenta', label='theory')

# it's nice if other people know what you did
plt.title('Bacteria Slowly Die Out (Experiment #17)')
plt.ylabel('number of bacteria (normalized)')
plt.xlabel('time (hours)')
plt.legend()
plt.grid()

# I can hardly wait.
plt.show()
