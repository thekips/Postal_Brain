import os
from typing import List
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from utils.typedef import Point

CWD = os.path.dirname(__file__) + '/'

class Image(object):
    def __init__(self, agent_loc: Point, object_loc: List[Point]) -> None:
        super().__init__()
        self.agent_x = agent_loc[0]
        self.agent_y = agent_loc[1]
        self.object_x = [value[0] for value in object_loc]
        self.object_y = [value[1] for value in object_loc]

        self.img = np.ndarray

    def __genBackground(self, isplot=False):
        # TODO(thekips): so slow that only read from old img.
        sns.kdeplot(x=self.object_x, y=self.object_y, shade=True)
        plt.axis('off')
        if isplot: plt.show()
        plt.savefig(CWD + 'res/current.jpg', dpi=1, bbox_inches='tight')
        plt.clf()

    def getBackground(self, isplot=False):
        self.img = plt.imread(CWD + 'res/current.jpg') 
        return self.img
    
    def getImage(self, isplot=False):
        self.img = plt.imread(CWD + 'res/current.jpg') 
        plt.scatter(x=self.agent_x, y=self.agent_y, c='red')

        return

    def getHeatMap(self, isplot=False):
        # TODO(thekips): so slow that only read from old img.
        # plt.scatter(x=self.object_x, y=self.object_y, c='blue')
        # plt.scatter(x=self.agent_x, y=self.agent_y, c='red')
        # plt.axis('off')
        # if isplot: plt.show()
        # plt.savefig(CWD + 'res/current.jpg', dpi=100)
        # plt.clf()

        self.img = plt.imread(CWD + 'res/current.jpg')
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