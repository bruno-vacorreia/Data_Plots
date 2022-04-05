import matplotlib.pyplot as plt

year = list(range(1990, 2000))
pop = [2.5, 3.5, 4.6, 4.9, 5.3, 5.9, 6.9, 7.8, 8.4, 9.2]

values = [0.01, 0.5, 3.7, 5.0, 4.2, 2.7, 4.3, 8.3]

value = values.extend(2)

x = [1, 2, 3, 4]

plt.plot(year, pop)                     # Curve
# plt.scatter(year, pop)                # Points
# plt.hist(values, bins=3)              # Histogram

plt.xlabel('Year', fontsize=18)                      # X axis label
plt.ylabel('Population (Billions)', fontsize=18)     # Y axis label
plt.title('World population', fontsize=30, fontweight='bold')           # Graph title
plt.yticks(range(0, 11))                # Points in Y axis
plt.xticks(year)                        # Points in X axis

plt.show()
