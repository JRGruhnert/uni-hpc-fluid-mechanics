
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

nx = 15
ny = 10

C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
F = np.random.normal(0, 0.1, (9, nx, ny))


def streaming(f, c):
    for i in range(9):
        f[i] = np.roll(f[i], c[i], axis=(0, 1))
    return f

def calculate_density(f):
    return np.sum(f, axis=0)

def calculate_velocity_field(f, density, c):
    return np.dot(f.T, c).T / density
   

streaming(F, C)
density = calculate_density(F)
velocity_field = calculate_velocity_field(F, density, C)

# Visualize density and velocity field
X, Y = np.meshgrid(range(nx), range(ny))

plt.figure(figsize=(10, 10))
plt.imshow(density.T, cmap=plt.cm.Reds)
plt.colorbar()
plt.quiver(X, Y, velocity_field[0].T, velocity_field[1].T)
plt.show()


