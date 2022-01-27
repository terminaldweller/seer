#!/usr/bin/env python3
import time
import random


class Particle:
    def __init__(self, x, y, z):
        self.Pbest_x = 0
        self.Pbest_y = 0
        self.Pbest_z = 0
        self.x = x
        self.y = y
        self.z = z
        self.x_k1 = 0
        self.y_k1 = 0
        self.z_k1 = 0
        self.V_x = 0
        self.V_y = 0
        self.V_z = 0
        self.V_xk1 = 0
        self.V_yk1 = 0
        self.V_zk1 = 0

    def __repr__(self):
        return (
            "{x: %s , y: %s, z: %s, Pbest_x: %s, Pbest_y: %s, Pbest_z: %s}\n"
            % (
                self.x,
                self.y,
                self.z,
                self.Pbest_x,
                self.Pbest_y,
                self.Pbest_z,
            )
        )

    def __str__(self):
        return (
            "{x: %s , y: %s, z: %s, Pbest_x: %s, Pbest_y: %s, Pbest_z: %s}\n"
            % (
                self.x,
                self.y,
                self.z,
                self.Pbest_x,
                self.Pbest_y,
                self.Pbest_z,
            )
        )

    def update_X(self):
        self.x = self.x + self.V_xk1
        self.y = self.y + self.V_yk1
        self.z = self.z + self.V_zk1

    def update_V(self, w, c1, c2, Gbest_x, Gbest_y, Gbest_z):
        rand1 = random.random()
        rand2 = random.random()
        self.V_xk1 = (
            w * self.V_x
            + c1 * rand1 * (self.Pbest_x - self.x)
            + c2 * rand2 * (Gbest_x - self.x)
        )
        self.V_yk1 = (
            w * self.V_y
            + c1 * rand1 * (self.Pbest_y - self.y)
            + c2 * rand2 * (Gbest_y - self.y)
        )
        self.V_zk1 = (
            w * self.V_z
            + c1 * rand1 * (self.Pbest_z - self.z)
            + c2 * rand2 * (Gbest_z - self.z)
        )

    def update_Pbest(self, x, y, z: float):
        self.Pbest_x = x
        self.Pbest_y = y
        self.Pbest_z = z

    def doRound(self, w, c1, c2, Gbest_x, Gbest_y, Gbest_z, fitness):
        fitness_x = fitness(self.x, self.y, self.z)
        self.update_V(w, c1, c2, Gbest_x, Gbest_y, Gbest_z)
        self.update_X()
        if abs(fitness(self.Pbest_x, self.Pbest_y, self.Pbest_z)) > abs(
            fitness_x
        ):
            self.update_Pbest(self.x, self.y, self.z)


class PSO:
    def __init__(self, w, c1, c2, particle_count):
        self.Gbest_x = 0
        self.Gbest_y = 0
        self.Gbest_z = 0
        self.particle_count = particle_count
        self.Particles = self.factory()
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def factory(self):
        result = list()
        for _ in range(1, self.particle_count):
            x = (
                random.random() * 10
                if random.random() > 0.5
                else -random.random() * 10
            )
            y = (
                random.random() * 10
                if random.random() > 0.5
                else -random.random() * 10
            )
            z = (
                random.random() * 10
                if random.random() > 0.5
                else -random.random() * 10
            )
            result.append(Particle(x, y, z))
        return result

    def fitness(self, x, y, z: float):
        return (
            (x ** 5) - ((x ** 2) * y * z) + (z * x) + (y ** 2) - (z ** 3) - 10
        )

    def doRround(self):
        roundBest_x = float()
        roundBest_y = float()
        roundBest_z = float()
        for particle in self.Particles:
            if abs(self.fitness(roundBest_x, roundBest_y, roundBest_z)) > abs(
                self.fitness(particle.x, particle.y, particle.z)
            ):
                roundBest_x = particle.x
                roundBest_y = particle.y
                roundBest_z = particle.z
        self.Gbest_x = roundBest_x
        self.Gbest_y = roundBest_y
        self.Gbest_z = roundBest_z
        for particle in self.Particles:
            particle.doRound(
                self.w,
                self.c1,
                self.c2,
                self.Gbest_x,
                self.Gbest_y,
                self.Gbest_z,
                self.fitness,
            )

    def printGlobalBest(self):
        print(
            "x: %s, y: %s, z: %s, fitness: %s"
            % (
                self.Gbest_x,
                self.Gbest_y,
                self.Gbest_z,
                self.fitness(self.Gbest_x, self.Gbest_y, self.Gbest_z),
            ),
        )


def main():
    random.seed(time.time())
    round_count = 10
    pso = PSO(5, 1.5, 1.5, 50)
    for _ in range(1, round_count):
        pso.doRround()
    pso.printGlobalBest()


if __name__ == "__main__":
    main()
