#!/usr/bin/env python3
import math
import time
import random


Nodes = [
    [0, 10, 20, 30, 40],
    [10, 0, 20, 30, 40],
    [20, 20, 0, 30, 40],
    [30, 30, 30, 0, 40],
    [40, 40, 40, 40, 0],
]

Pheromone = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]


class Ant:
    def __init__(self, node):
        self.node = node
        self.route = int()
        self.exclusion_list = list()

    def __repr__(self):
        return "{node: %s}" % (self.node)

    def __str__(self):
        return "{node: %s}" % (self.node)

    def nextHop(self):
        n = math.ceil(random.random() * len(Nodes)) - 1
        if (
            n not in self.exclusion_list
            and n != self.exclusion_list[-1]
            and Nodes[self.exclusion_list[-1]][n] != 0
        ):
            self.exclusion_list.append(n)
            return n
        else:
            if len(self.exclusion_list) == len(Nodes):
                return None
            else:
                self.nextHop()

    def nextHop2(self):
        pass

    def walk(self):
        self.exclusion_list.append(self.node)
        for _ in range(0, len(Nodes)):
            next = self.nextHop()
            if next is not None:
                self.route += Nodes[self.node][next]
                self.node = next

    def printRoute(self):
        print(self.exclusion_list)


class ACO:
    def __init__(self, ant_count):
        self.ant_count = ant_count
        self.Ants = self.factory()

    def factory(self):
        result = list()
        for _ in range(0, self.ant_count):
            node = math.ceil(random.random() * 5) - 1
            result.append(Ant(node))
        return result

    def fitness(self):
        pass

    def run(self):
        for ant in self.Ants:
            ant.walk()
            ant.printRoute()
            for i in range(0, len(Nodes) - 1):
                Pheromone[ant.exclusion_list[i]][ant.exclusion_list[i + 1]] = (
                    1 / ant.route
                )


def main():
    random.seed(time.time())
    rho = 0
    alpha = 1
    beta = 1
    aco = ACO(5)
    # for ant in aco.Ants:
    #     print("ant:", ant)
    aco.run()
    for i in range(0, len(Pheromone)):
        for j in range(0, len(Pheromone)):
            print(Pheromone[i][j], end=" ")
        print()


if __name__ == "__main__":
    main()
