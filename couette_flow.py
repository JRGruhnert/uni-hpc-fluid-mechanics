import os
import numpy as np
import matplotlib.pyplot as plt
from lb import LatticeBoltzmann
from plot import Plotter
from lb import RigidWall, MovingWall, PeriodicWall

def couette_flow_sim(nx: int = 100, ny: int = 100, omega: float = 0.6, epsilon: float = 0.01,
                          steps: int = 200, p0: float = 1.0):
    
    # split velocity set into x and y components (unused right now)
    #ux = np.zeros((nx, ny))
    #uy = np.zeros((nx, ny))
    #x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    rho = np.zeros((nx, ny))
    velocities = np.zeros((2, nx, ny))
    boundaries = [MovingWall("top"), RigidWall("bottom"), PeriodicWall("left"), PeriodicWall("right")]


    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter()

    for(step) in range(steps):
        latticeBoltzmann.tick()
        rho, velocities = latticeBoltzmann.rho, latticeBoltzmann.velocities
        plotter.plot_shear_wave(velocities, step, epsilon, nx, ny)


