import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from random import random

class CAMPO:
    def __init__(self, grid_size=(50, 50), initial_density=0.1, dispersal_prob=0.1, extinction_prob=0.05, iterations=100):
        self.grid_size = grid_size
        self.initial_density = initial_density
        self.dispersal_prob = dispersal_prob
        self.extinction_prob = extinction_prob
        self.iterations = iterations
        self.grid = np.zeros(grid_size)
        self.initialize_grid()

    def initialize_grid(self):
        num_patches = np.prod(self.grid_size)
        num_initial = int(self.initial_density * num_patches)
        positions = [(randint(0, self.grid_size[0]-1), randint(0, self.grid_size[1]-1)) for _ in range(num_initial)]
        for pos in positions:
            self.grid[pos] = 1

    def update_grid(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] == 1:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                                if random() < self.dispersal_prob:
                                    new_grid[ni, nj] = 1
                    if random() < self.extinction_prob:
                        new_grid[i, j] = 0
                else:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                                if self.grid[ni, nj] == 1 and random() < self.dispersal_prob:
                                    new_grid[i, j] = 1

    def run_simulation(self):
        for t in range(self.iterations):
            self.update_grid()

    def plot_simulation(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.grid, cmap='Greens', interpolation='nearest')
        plt.title('CAMPO Simulation')
        plt.colorbar(label='Species Presence')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

campo = CAMPO(grid_size=(50, 50), initial_density=0.1, dispersal_prob=0.1, extinction_prob=0.05, iterations=100)
campo.run_simulation()
campo.plot_simulation()

def func(x, y):
    res = 4*x**2 - 2.1*x**4 + x**6/3 + x*y - 4*y**2 + 4*y**4
    return res

class SA:
    def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = func
        self.iter = iter
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0
        self.x = [random() * 11 - 5 for i in range(iter)]
        self.y = [random() * 11 - 5 for i in range(iter)]
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) and (-5 <= y_new <= 5):
                break
        return x_new, y_new

    def Metropolis(self, f, f_new):
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):
        f_list = [self.func(self.x[i], self.y[i]) for i in range(self.iter)]
        f_best = min(f_list)
        idx = f_list.index(f_best)
        return f_best, idx

    def run(self):
        while self.T > self.Tf:
            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])
                x_new, y_new = self.generate_new(self.x[i], self.y[i])
                f_new = self.func(x_new, y_new)
                if self.Metropolis(f, f_new):
                    self.x[i] = x_new
                    self.y[i] = y_new
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            self.T = self.T * self.alpha

        f_best, idx = self.best()
        print(f"最优解: F={f_best}, x={self.x[idx]}, y={self.y[idx]}")

        plt.plot(self.history['T'], self.history['f'])
        plt.xlabel('Temperature')
        plt.ylabel('Objective Function Value')
        plt.gca().invert_xaxis()
        plt.show()

sa = SA(func)
sa.run()
