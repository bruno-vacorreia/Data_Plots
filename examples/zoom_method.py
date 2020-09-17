import matplotlib.pyplot as plt
import numpy as np

step = 0.1
x = np.arange(0, 10 + step, step)
y = x**3

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y)
axins = ax.inset_axes([0.1, 0.5, 0.4, 0.4])
axins.plot(x[:10], y[:10])
ax.indicate_inset_zoom(axins, linewidth=3)
axins.set_xticklabels('')
axins.set_yticklabels('')
plt.show()
