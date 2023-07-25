import numpy as np
from lb import LatticeBoltzmann, RigidWall, TopMovingWall
from plot import Plotter4

def sliding_lid_sim(nx: int = 50, ny: int = 50, omega: float = 0.3, steps: int = 200001):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    wall_velocity = 0.05
    boundaries = [RigidWall("bottom"), TopMovingWall("top", wall_velocity), RigidWall("left"), RigidWall("right")]

    #omega = 1 / (0.5 + ((wall_velocity * nx) / 1000) / (1/3))
    omega = 1.0
    assert(omega <= 1.7)
    
    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter4()

    for(step) in range(steps):
        latticeBoltzmann.tick()
        
        # plot sliding lid every 1000 steps 
        if((step % 50000 == 0)):
            plotter.plot_sliding_lid(latticeBoltzmann.velocities, step, nx, ny)
            print("Step: {}".format(step))

sliding_lid_sim()
