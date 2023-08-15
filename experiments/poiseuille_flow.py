import numpy as np
from lb.boundaries import Periodic, RigidWall
from lb.lattice_boltzmann import LatticeBoltzmann
from visualisation.plot import PoiseuilleFlowPlotter

def poiseuille_flow_sim(nx: int, ny: int, total_steps: int, plot_every: int, output_dir: str, omega: float, pressure_in: float, pressure_out: float):
    
    rho = np.ones((nx, ny)) #rho with ghost cells (inlet and outlet)
    velocities = np.zeros((2, nx, ny)) #velocities with ghost cells (inlet and outlet)
    
    boundaries = [Periodic("left", ny, pressure_in), Periodic("right", ny, pressure_out), RigidWall("top"), RigidWall("bottom")]

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = PoiseuilleFlowPlotter(nx, ny, total_steps, output_dir, omega, pressure_in, pressure_out)

    for(step) in range(total_steps):
        latticeBoltzmann.tick()
        
        if((step % plot_every == 0)):
            plotter.plot(latticeBoltzmann.rho, latticeBoltzmann.velocities, step)
            print("Step: {}".format(step))
