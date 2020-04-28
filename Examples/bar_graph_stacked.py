"""
New Planet School

This Python script illustrates how to make bar graphs. The example 
used is U.S. pennies in circulation from the two mints. 
Data from:
    http://www.coinnews.net/mints/us-circulating-coin-production-figures/
"""

# these libraries will be very useful, bring 'em in
import numpy as np
import matplotlib.pyplot as plt

# data is put into Python lists here
years = [1999, 2000, 2001, 2000, 2003, 2004, # use as x labels
         2005, 2006, 2007, 2008, 2000, 2010,
         2011, 2012] 
Denver = [6360.07, 8774.22, 5374.99, 4028.06, 
          3548.00, 3456.40, 3764.45, 3944.00,
          3638.80, 2849.6, 1248.00, 2047.20,
          2536.14, 2883.20]
Philly = [5237.60, 5503.20, 4959.60, 3260.80, 3300.00, 
          3379.60, 3935.60, 4290.00, 3762.40,
          2569.60, 1106.00, 1963.63, 2402.40,
          3132.00]

# set up some values for the bars
bar_width = 0.4 # set the width of the bars
x = np.arange(len(years)) # need an array of x values (but, see below)
opacity = 0.5 # not so dark

# setup the plots: both points and smooth curve                        
plt.bar(x, Denver, bar_width, color='green', label='Denver', alpha=opacity)
plt.bar(x, Philly, bar_width, color='orange', label='Philadelphia', alpha=opacity, bottom=Denver)

# Setup the plot configuration
plt.legend()
plt.xlabel('year')
plt.ylabel('number of pennies in circulation (millions)')
plt.title('Pennies from Denver and Philadelphia Mints')
plt.xticks(x, years) # override the xlabels with our custom labels

# okay, I think we are all ready...
plt.tight_layout() # this helps spread things - easier to read
plt.show()
