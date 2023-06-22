from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from lb import LatticeBoltzmann


def shear_wave_sim(nx: int = 100, ny: int = 100, w: float = 1.0, eps: float = 0.01,
                          steps: int = 100, p0: float = 1.0):
    
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    density = p0 + eps * np.sin(2*np.pi/nx*x)
    velocity_field = np.zeros((2, nx, ny))
    latticeBoltzmann = LatticeBoltzmann(density, velocity_field)

    for(i) in range(steps):
        latticeBoltzmann.tick()
        density, velocity_field = latticeBoltzmann.output()
        plt.imshow(np.sqrt(velocity_field[0]**2 + velocity_field[1]**2))
        plt.pause(0.1)
        plt.cla()


shear_wave_sim()