from matplotlib import pyplot as plt
from lb import LatticeBoltzmann
import numpy as np


nx = 15
ny = 10
# Visualize density and velocity field
X, Y = np.meshgrid(range(nx), range(ny))

c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

f = np.ones((ny, nx, 9)) + 0.01 * np.random.randn(ny, nx, 9)
f[:, :, 3] = 2.3
x, y = np.meshgrid(np.arange(nx), np.arange(ny))

density = 1.0 + 0.01 * np.sin(2*np.pi/nx*x)
velocity_field = np.zeros((ny, nx, 2), dtype=np.float32)


class Simulation():
    def __init__(self):
        self.latticeBoltzmann = LatticeBoltzmann(f, density, velocity_field)
    
    def run(self, ticks):
        for(i) in range(ticks):
            self.latticeBoltzmann.tick()
            density, velocity_field = self.latticeBoltzmann.output()
            plt.imshow(np.sqrt(velocity_field[0]**2 + velocity_field[1]**2))
            plt.pause(0.1)
            plt.cla()