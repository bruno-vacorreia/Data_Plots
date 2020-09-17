import matplotlib.pyplot as plt
import numpy as np
import random

X = list(range(10))
Y = [x+(x*random.random()) for x in X]

fig = plt.figure(figsize=(12,6))

plt.plot(X, np.exp(X))
plt.title('Annotation Trick')
plt.annotate('Point 1', xy=(6, 400), arrowprops=dict(arrowstyle='->'), xytext=(4, 600))
plt.annotate('Point 2', xy=(7, 1150), arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-.2'),
             xytext=(4.5, 2000))
plt.annotate('Point 3', xy=(8, 3000), arrowprops=dict(arrowstyle='-|>', connectionstyle='angle, angleA=90, angleB=0'),
             xytext=(8.5, 2200))

plt.show()
