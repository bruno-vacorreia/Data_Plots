"""
New
    Planet
        School

Explore multiple plots using the subplot capability. Use oscillator
damping as the example (e.g., circuits, spring, mechanical vibration, etc.). 
"""

# I suppose we now know to do this....
import numpy as np
import matplotlib.pyplot as plt

# define funtions
t = np.linspace(0.0, 8.0, 80)
y_ud = np.exp(-t)*np.cos(2.*t) # under damped
y_cd = (1 + t)*np.exp(-t)      # critically damped
y_od = np.exp(-t)              # over damped

# First plot
plt.subplot(311)
plt.plot(t, y_ud, lw=2, color='magenta', label='under damped')
plt.xlim(0, 8)
plt.legend()
plt.grid()

# Second plot
plt.subplot(312)
plt.plot(t, y_cd, lw=2, color='red', label='critically damped')
plt.xlim(0, 8)
plt.legend()
plt.grid()
plt.ylabel('amplitude (meters)') # notice where I put this!

# Third plot
plt.subplot(313)
plt.plot(t, y_od, lw=2, color='orange', label='over damped')
plt.xlim(0, 8)
plt.legend()
plt.grid()
plt.xlabel('time (seconds)') # notice where I put this!

# looks like are all ready with the set up; we can see why "deferred
# rendering" is so useful
plt.show()

