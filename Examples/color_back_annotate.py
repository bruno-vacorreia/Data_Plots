"""
New Planet School

Example to show colored backgrounds and grids. 

And, add to that text on the graph, larger numbers, and an arrow.
"""

# yep, gonna need these very nice modules
import numpy as np
import matplotlib.pyplot as plt

# set up points along abscissa
x = np.linspace(0, 2*np.pi, 100)
y = np.cos(2.5*x)*np.exp(-x/2.)

# set up coarse grid and function for comparison
xc = np.linspace(0, 2*np.pi, 10)
yc = np.cos(2.5*xc)*np.exp(-xc/2.)

# change color of background
plt.subplot(111, facecolor='slategrey')

# starting adding to the plot
plt.plot(x, y, lw=3, color='blue')
plt.plot(xc, yc, 'yo--', lw=2)

# make the plot useful to other people
plt.ylabel('decaying signal (volts)', fontsize=20)
plt.xlabel('time (seconds)', fontsize=20)
plt.title('Calculus: Approximating A Function By Series Of Lines', fontsize=24)
plt.xlim(0, 2*np.pi)
plt.ylim(-0.65, 1.05)
plt.grid(color='white')

# add text to the plot
plt.text(1.5, 0.55, 'each line approximates\n the local slope\n of the curve', size=18)

# increase size of numbers on axes
plt.xticks(size=16)
plt.yticks(size=16)

# add an arrow
# the format is plt.arrow(x, y, del_x, del_y, ...)
plt.arrow(2.6, 0.58, 1.25, -0.65, shape='full', lw=3)

# Is everybody ready to see it? I am!
plt.show()
