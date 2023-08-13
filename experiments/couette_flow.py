import numpy as np
from lb.boundaries import RigidWall, TopMovingWall
from lb.lattice_boltzmann import LatticeBoltzmann

from visualisation.plot import Plotter2

def couette_flow_sim(nx: int, ny: int, total_steps: int, plot_every: int, output_dir: str, omega: float, wall_velocity: float):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    boundaries = [TopMovingWall("top", wall_velocity), RigidWall("bottom")]

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter2(nx, ny, total_steps, plot_every, wall_velocity, output_dir)

    for(step) in range(total_steps):
        latticeBoltzmann.tick()
        
        # plot every 200 steps 
        if((step % plot_every == 0)):
            plotter.plot_cuette_flow(step, latticeBoltzmann.velocities)
            print("Step: {}".format(step))
    plotter.save(total_steps -1)
