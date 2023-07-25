import numpy as np
from lb import LatticeBoltzmann, RigidWall, TopMovingWall
from plot import Plotter2

def couette_flow_sim(nx: int = 50, ny: int = 50, omega: float = 0.3, steps: int = 3001):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    wall_velocity = 0.1
    boundaries = [TopMovingWall("top", wall_velocity), RigidWall("bottom")]

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter2(nx, ny, wall_velocity)

    for(step) in range(steps):
        latticeBoltzmann.tick()
        
        # plot every 200 steps 
        if((step % 200 == 0)):
            plotter.plot_cuette_flow(latticeBoltzmann.velocities)
            print("Step: {}".format(step))
    plotter.save(steps -1)
couette_flow_sim()
