"""
New Planet School

This Python script illustrates how to:
    * make fake data
    * plot data points and functions on the same graph
    * use LOG scales
This is done in the context of exponential decay.
"""

# these libraries will be very useful, bring 'em in
import numpy as np
import matplotlib.pyplot as plt

# make fake data by adding random values to a function
x_fake = np.linspace(0, 5, 40) # points where data was taken
noise = 0.025*np.random.normal(size=len(x_fake)) # random numbers
y_fake = np.exp(-x_fake) + noise # data is theory plus noise

# create theoretical curve to compare with "data"
x_theory = np.linspace(0, 5, 200) # use more values to get smooth curve
y_theory = np.exp(-x_theory)

# setup the plots: both points and smooth curve
plt.scatter(x_fake, y_fake, color='r', label='data', lw=1)                  # points
plt.plot(x_theory, y_theory, color='green', label='theory', lw=3)   # line
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('data and theory')
plt.title('Results From The Experiment')
plt.xticks(range(6))

# make the y axis (ordinate) log; that is, log-linear
plt.yscale('log')

# show me!
plt.show()
