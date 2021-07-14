import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas

x = np.random.random(100) + 112.5 
y = np.random.random(100) + 22.5

fig1 = plt.figure(1)
sns.kdeplot(x=x, y=y, shade=True)
img1 = np.array(fig1.canvas.get_renderer()._renderer)
print(img1.shape)

fig2 = plt.figure(2)
sns.scatterplot(x=x, y=y)
img2= np.array(fig1.canvas.get_renderer()._renderer)
print(img2.shape)

plt.show()
