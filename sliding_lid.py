import numpy as np
from lb import LatticeBoltzmann, RigidWall, MovingWall
from plot import Plotter4

def sliding_lid_sim(nx: int = 300, ny: int = 300, omega: float = 0.3, steps: int = 20001):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    wall_velocity = np.array([0.0, 0.1])
    boundaries = [MovingWall("top", wall_velocity, rho), RigidWall("bottom"), RigidWall("left"), RigidWall("right")]

    #omega = 1 / (0.5 + ((wall_velocity[1] * nx) / 1000) / (1/3))
    #assert(omega < 1.7)
    
    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter4()

    for(step) in range(steps):
        latticeBoltzmann.tick()
        
        # plot sliding lid every 200 steps 
        if((step % 1000 == 0)):
            plotter.plot_sliding_lid(latticeBoltzmann.velocities, step, nx, ny)
            print("Step: {}".format(step))

sliding_lid_sim()
