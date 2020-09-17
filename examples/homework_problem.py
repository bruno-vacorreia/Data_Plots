"""
New
Planet
School
"""

# import the usual
import matplotlib.pyplot as plt
import numpy as np

# open the file
my_file = open('/WhereDidYouSaveIt/SF_weather_2014.csv', 'r')

# need to skip the header
header = my_file.readline()
print(header)

# loop over lines in the file
max_temp = []
min_temp = []
day_of_year = []
day = 0
for lines in my_file:
    day += 1
    # split the line into separate strings
    x = lines.split(',')
    max_temp.append(float(x[1]))
    min_temp.append(float(x[2]))
    day_of_year.append(day)

# overlap a contour plot
x = range(365)
y = range(17, 99) # adjust contour to flit with data
z = [[z] * 365 for z in range(len(y))]
num_bars = 200  # more bars = smoother gradient
plt.contourf(x, y, z, num_bars)

plt.grid(alpha=0.5) # gentle grid
plt.plot(day_of_year, max_temp, label='high', lw=1, color='grey')
plt.plot(day_of_year, min_temp, label='low', lw=1, color='grey')
plt.xlabel('day of year', fontsize=20)
plt.ylabel('temperature (F)', fontsize=20)
plt.title('Santa Fe, New Mexico Temperatures 2014', fontsize=22)
plt.xlim(1, 365)
# cover up parts of contour plot that need to be white
plt.fill_between(day_of_year, 0.0, min_temp, alpha=0.99, color='white')
plt.fill_between(day_of_year, max_temp, 100.0, alpha=0.99, color='white')
# plt.legend()

# let's see it!
plt.show()

# always close the file when you are done!
my_file.close()

