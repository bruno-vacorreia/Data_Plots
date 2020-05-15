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
bar_width = 0.4             # set the width of the bars
y = np.arange(len(years))   # need an array of x values (but, see below)
opacity = 1.0               # not so dark
desl = bar_width / 2

# setup the plots: both points and smooth curve
plt.barh(y - desl, Denver, bar_width, color='green', label='Denver', alpha=opacity, zorder=2)
plt.barh(y + desl, Philly, bar_width, color='orange', label='Philadelphia', alpha=opacity, zorder=2)

# Setup the plot configuration
plt.grid(alpha=0.4, ls='dashed', axis='x', zorder=1)
plt.legend(fontsize=12)
plt.xlabel('number of pennies in circulation (millions)', fontsize=14)
plt.ylabel('year', fontsize=14)
plt.title('Pennies from Denver and Philadelphia Mints', fontsize=18, fontweight='bold')
plt.yticks(y, years)                    # override the xlabels with our custom labels
# plt.xticks(np.linspace(0, 9000, 10))
plt.xlim(0, 9000)
# plt.xlim(0, 2*np.pi)
plt.tight_layout() # this helps spread things - easier to read

# Plot figure
plt.show()
