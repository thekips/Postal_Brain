import json
import os
from typing import List
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


CWD = os.path.dirname(__file__) + '/'

class Image(object):
    def __init__(self, agent_loc, object_loc) -> None:
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

    def getHeatMap(self, color=False, isplot=False):
        # TODO(thekips): so slow that only read from old img.
        if color == False:
            plt.scatter(x=self.object_x, y=self.object_y, c='blue')
        else:
            plt.scatter(x=self.object_x, y=self.object_y, c=color)
        plt.scatter(x=self.agent_x, y=self.agent_y, c='red')
        plt.axis('off')
        if isplot: plt.show()
        plt.savefig(CWD + 'res/current.jpg', dpi=100)
        plt.clf()

        self.img = plt.imread(CWD + 'res/current.jpg')
        return self.img

class GridPic():
    def __init__(self) -> None:
        pass

    def from_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        grid_size = int(pow(len(data), 0.5))
        grid_list = [[0 for i in range(grid_size)] for j in range(grid_size)]
        max_value = 0
        max_i, max_j = 0, 0
        for i in range(grid_size):
            for j in range(grid_size):
                grid_list[i][j] = 8 / data[str((i, j))]
                if max_value < grid_list[i][j]:
                    max_value = grid_list[i][j]
                    max_i, max_j = i, j

        print(max_i, max_j, max_value)
        plt.cla()
        sns.heatmap(grid_list, cmap='Greys')
        # plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # # For test.
    # xx = [x // 10 for x in range(0, 100)]
    # yy = [x for x in range(10)] * 10
    # color = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    # object_loc = [*zip(xx, yy)] 
    # agent_loc = (3, 3)
    # image = Image(agent_loc, object_loc)
    # img = image.getHeatMap(color)
    # plt.imshow(img)
    # plt.show()

    x = GridPic()
    path = os.path.dirname(__file__) + '/dist/true_opt_value.json'
    x.from_json(path)