import math
import numpy as np
import matplotlib.pyplot as plt
from random import random, uniform, randint

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

    def run_simulation(self, l, w, h):
        for t in range(self.iterations):
            self.update_grid(l, w, h)

    def plot_simulation(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.grid, cmap='Greens', interpolation='nearest')
        plt.title('CAMPO Simulation')
        plt.colorbar(label='Species Presence')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

class SolarConcentratorSA:
    def __init__(self, campo_simulator, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.campo = campo_simulator
        self.iter = iter
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0
        self.l = uniform(2, 8)
        self.w = uniform(2, 6)
        self.h = uniform(5, 10)
        self.best_params = (self.l, self.w, self.h)
        self.best_objective = self.objective_function(self.l, self.w, self.h)
        self.history = {'objective': [], 'T': []}

    def objective_function(self, l, w, h):
        concentration_efficiency = l * w / h
        return concentration_efficiency

    def generate_new(self, l, w, h):
        l_new = max(2, min(8, l + uniform(-0.5, 0.5)))
        w_new = max(2, min(6, w + uniform(-0.5, 0.5)))
        h_new = max(5, min(10, h + uniform(-1, 1)))
        return l_new, w_new, h_new

    def Metropolis(self, f_current, f_new):
        if f_new > f_current:
            return 1
        else:
            p = math.exp((f_new - f_current) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def run(self):
        while self.T > self.Tf:
            for _ in range(self.iter):
                l_new, w_new, h_new = self.generate_new(self.l, self.w, self.h)
                f_current = self.objective_function(self.l, self.w, self.h)
                f_new = self.objective_function(l_new, w_new, h_new)
                if self.Metropolis(f_current, f_new):
                    self.l, self.w, self.h = l_new, w_new, h_new
                    self.best_params = (self.l, self.w, self.h)
                    self.best_objective = f_new
            self.history['objective'].append(self.best_objective)
            self.history['T'].append(self.T)
            self.T *= self.alpha

        self.campo.run_simulation(self.l, self.w, self.h)
        self.campo.plot_simulation()

        print(f"Best parameters: l={self.best_params[0]}, w={self.best_params[1]}, h={self.best_params[2]}")
        print(f"Best objective value: {self.best_objective}")

        plt.plot(self.history['T'], self.history['objective'])
        plt.title('Simulated Annealing Optimization')
        plt.xlabel('Temperature')
        plt.ylabel('Objective Value')
        plt.gca().invert_xaxis()  # Invert x-axis to show decreasing temperature
        plt.show()

campo = CAMPO(grid_size=(50, 50), initial_density=0.1, dispersal_prob=0.1, extinction_prob=0.05, iterations=100)
sa = SolarConcentratorSA(campo_simulator=campo)
sa.run()
