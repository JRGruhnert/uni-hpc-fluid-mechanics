import numpy as np
from lb.boundaries import TopMovingWall, RigidWall
from lb.lattice_boltzmann import *
from lb.vars import CS
from visualisation.plot import SlidingLidPlotter

def sliding_lid_sim(nx: int, ny: int, total_steps: int, plot_every: int, output_dir: str, reynolds: int, wall_velocity: float):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    omega = 1 / (0.5 + ((wall_velocity * nx) / reynolds) / (1/3))
    viscosity = CS**2 * (1.0 / omega - 0.5) # for Re=1000
    reynolds_number = wall_velocity * nx / viscosity  # calculate reynolds number
    print("Reynolds number: {}".format(reynolds_number))
    print("Omega: {}".format(omega))
    print("Viscosity: {}".format(viscosity))
    print("Wall velocity: {}".format(wall_velocity))
    boundaries = [TopMovingWall("top", wall_velocity), RigidWall("bottom"), RigidWall("left"), RigidWall("right")]

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = SlidingLidPlotter(nx, ny, total_steps, output_dir, viscosity, wall_velocity)

    for(step) in range(total_steps):
        # perform one step of the simulation
        latticeBoltzmann.tick()
        
        # plot sliding lid every 1000 steps 
        if((step % plot_every == 0)):
            plotter.plot(latticeBoltzmann.velocities, step)
            print("Step: {}".format(step))

