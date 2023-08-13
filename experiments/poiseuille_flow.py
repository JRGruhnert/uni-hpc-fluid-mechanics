import numpy as np
from lb.boundaries import Periodic, RigidWall
from lb.lattice_boltzmann import LatticeBoltzmann
from visualisation.plot import PoiseuilleFlowPlotter

def poiseuille_flow_sim(nx: int, ny: int, total_steps: int, plot_every: int, output_dir: str, omega: float, pressure_left: float, pressure_right: float):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    
    boundaries = [Periodic("left", ny, pressure_left), Periodic("right", ny, pressure_right), RigidWall("top"), RigidWall("bottom")]

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = PoiseuilleFlowPlotter(nx, ny, total_steps, output_dir, omega, pressure_left, pressure_right)

    for(step) in range(total_steps):
        latticeBoltzmann.tick()
        
        if((step % plot_every == 0)):
            plotter.plot(latticeBoltzmann.rho, latticeBoltzmann.velocities, step)
            print("Step: {}".format(step))
