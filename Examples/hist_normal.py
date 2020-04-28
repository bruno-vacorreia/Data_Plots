"""
New Planet School

This histogram example uses a random number
generator from NumPy that generates numbers from
a "normal" distribution function.
"""

# We just might be needing these....
import numpy as np
import matplotlib.pyplot as plt

# store data in NumPy arrays for later use
rand_num = np.random.randn(100000)

# set up histograms
plt.hist(rand_num, bins=100, alpha=0.4, label='bell curve', color='orange')

# make graph readable; your teacher might want to see this!
plt.xlabel('random value')
plt.ylabel('distribution')
plt.title('Explore Statistics: Normal Distribution With 10000 Values')
plt.legend(loc='upper left') # move it over so that it doesn't cover anything

# all done with the set up; show us!
plt.show()

