#!/usr/bin/env python3
from random import seed, random
from time import time
import math


class X:
    def __init__(self, x: float, y: float, z: float):
        self.x_now = 0
        self.y_now = 0
        self.z_now = 0
        self.x_k = x
        self.y_k = y
        self.z_k = z
        self.x_k1 = 0
        self.y_k1 = 0
        self.z_k1 = 0
        self.k = 0

    def neighbour(self):
        self.X_k_to_now()
        self.x_now = self.x_now + (
            random() / 100 if random() > 0.5 else -random() / 100
        )
        self.y_now = self.y_now + (
            random() / 100 if random() > 0.5 else -random() / 100
        )
        self.z_now = self.z_now + (
            random() / 100 if random() > 0.5 else -random() / 100
        )
        self.k = self.k + 1

    def function(self, x: float, y: float, z: float):
        return (
            (x ** 5) - ((x ** 2) * y * z) + (z * x) + (y ** 2) - (z ** 3) - 10
        )

    def X_now(self):
        return self.function(self.x_now, self.y_now, self.z_now)

    def X_k(self):
        return self.function(self.x_k, self.y_k, self.z_k)

    def X_k_to_now(self):
        self.x_now = self.x_k
        self.y_now = self.y_k
        self.z_now = self.z_k

    def X_now_to_k1(self):
        self.x_k1 = self.x_now
        self.y_k1 = self.y_now
        self.z_k1 = self.z_now

    def X_k_to_k1(self):
        self.x_k1 = self.x_k
        self.y_k1 = self.y_k
        self.z_k1 = self.z_k

    def X_k1_to_k(self):
        self.x_k = self.x_k1
        self.y_k = self.y_k1
        self.z_k = self.z_k1


def SA():
    K_max = 600
    T_zero = 30
    alpha = 0.90
    alpha_step = 20
    x = X(1, 0, -1)
    T = T_zero
    seed(time())
    for k in range(1, K_max):
        x.neighbour()
        if x.X_now() <= x.X_k():
            x.X_now_to_k1()
        else:
            p = math.e ** ((-(x.X_now() - x.X_k())) / T)
            seed(time())
            r = random()
            if p >= r:
                x.X_now_to_k1()
            else:
                x.X_k_to_k1()
        if k % alpha_step == 0:
            T = T * alpha
        x.X_k1_to_k()
    print(x.x_k, x.y_k, x.z_k)
    print("k=", x.X_k())


def main():
    SA()


if __name__ == "__main__":
    main()
