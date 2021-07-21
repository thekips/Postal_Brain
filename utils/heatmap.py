from typing import List
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from utils.typedef import Point

class Image(object):
    def __init__(self, agent_loc: Point, object_loc: List[Point]) -> None:
        super().__init__()
        self.agent_x = agent_loc[0]
        self.agent_y = agent_loc[1]
        self.object_x = [value[0] for value in object_loc]
        self.object_y = [value[1] for value in object_loc]

        self.img = np.ndarray

    def getHeatMap(self, isplot=False):
        sns.kdeplot(x=self.object_x, y=self.object_y, shade=True)
        plt.scatter(x=self.agent_x, y=self.agent_y, c='red')
        plt.axis('off')
        if isplot: plt.show()
        plt.savefig('utils/res/current.jpg')
        plt.clf()

        self.img = plt.imread('utils/res/current.jpg') 
        return self.img

    def getScatterMap(self, isplot=False):
        plt.scatter(x=self.object_x, y=self.object_y, c='blue')
        plt.scatter(x=self.agent_x, y=self.agent_y, c='red')
        plt.axis('off')
        if isplot: plt.show()
        plt.savefig('utils/res/current.jpg')
        plt.clf()

        self.img = plt.imread('utils/res/current.jpg')
        return self.img

if __name__ == "__main__":
    x = np.random.random(100) + 112.5 
    y = np.random.random(100) + 22.5
    object_loc = [*zip(x, y)] 
    agent_loc = [113, 23]
    image = Image(agent_loc, object_loc)
    img = image.getScatterMap()
    plt.imshow(img)
    plt.show()