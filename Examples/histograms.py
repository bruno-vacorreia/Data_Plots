"""
New Planet School

This histogram example uses December 2014 daily high
and low temperatures in Santa Fe, New Mexico, USA. This 
data represents actual daily temperatures from the website
Weather Underground.
"""

# We just might be needing these....
import numpy as np
import matplotlib.pyplot as plt

# store data in NumPy arrays for later use
# (could also read data from a file using 'open' or 'np.loadtxt')
sf_high_temp = np.array([55, 52, 53, 46, 52, 51, 55, 55, 52, 
                         55, 57, 55, 55, 43, 42, 39, 34, 39, 
                         39, 39, 45, 52, 35, 36, 45, 30, 30, 34, 37, 21, 25])
sf_low_temp = np.array([23, 28, 28, 36, 33, 32, 32, 30, 32, 
                        28, 30, 32, 30, 25, 23, 25, 26, 19, 
                        21, 21, 32, 28, 18, 14, 25, 10, 3, 12, 9, 10, 8])

# set up histograms
plt.hist(sf_low_temp, bins=10, alpha=0.4, label='low')
plt.hist(sf_high_temp, bins=10, alpha=0.4, label='high')

# make graph readable;
plt.xlabel('temperatures (F)')
plt.ylabel('distribution')
plt.title('Santa Fe Low and High Temperatures for December 2014')
plt.legend(loc='upper left') # move it over so that it doesn't cover anything
plt.xticks(np.linspace(0, 60, 13))

# all done with the set up; show us!
plt.show()

