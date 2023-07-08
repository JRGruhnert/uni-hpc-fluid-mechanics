import os
import numpy as np
import matplotlib.pyplot as plt
from lb import LatticeBoltzmann
from plot import Plotter2
from lb import RigidWall, MovingWall, PeriodicWall

def couette_flow_sim(nx: int = 100, ny: int = 100, omega: float = 0.6, epsilon: float = 0.01,
                          steps: int = 100, p0: float = 1.0):
    
    # split velocity set into x and y components (unused right now)
    #ux = np.zeros((nx, ny))
    #uy = np.zeros((nx, ny))
    #x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    boundaries = [MovingWall([0.0, 0.1],"top"), RigidWall("bottom")]#, PeriodicWall("left"), PeriodicWall("right")]


    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter2()

    for(step) in range(steps):
        latticeBoltzmann.tick()
        rho, velocities = latticeBoltzmann.get_rho(), latticeBoltzmann.get_velocities()
        # plot shear wave every 200 steps 
        
        if((step % 200 == 0)):
            for boundary in boundaries:
                boundary.update_velocity(velocities)
            plotter.plot_cuette_flow(velocities, np.array(0.0, 0.1), step, nx, ny)
        print("Step: {}".format(step))

couette_flow_sim()
