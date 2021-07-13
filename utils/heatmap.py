import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random

from seaborn.distributions import histplot

x = np.random.random(100) + 112.5 
y = np.random.random(100) + 22.5

plt.scatter(x, y, c=(1, 0, 0))
plt.show()