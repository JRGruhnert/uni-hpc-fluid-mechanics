import numpy as np
from lb import LatticeBoltzmann, RigidWall, PortalWall
from plot import Plotter3

def poiseuille_flow_sim(nx: int = 100, ny: int = 50, pressure_left= 0.31, pressure_right= 0.3, omega: float = 0.3, steps: int = 4001):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    
    boundaries = [PortalWall("left", ny, pressure_left), PortalWall("right", ny, pressure_right), RigidWall("top"), RigidWall("bottom")]

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter3()

    for(step) in range(steps):
        latticeBoltzmann.tick()
        
        # plot every 200 steps 
        if((step % 200 == 0)):
            #for boundary in latticeBoltzmann.boundaries:
            #    boundary.update_velocity(latticeBoltzmann.velocities)

            plotter.plot_poiseuille_flow(latticeBoltzmann.rho, latticeBoltzmann.velocities, omega, pressure_left, pressure_right, step, nx, ny)
            print("Step: {}".format(step))

poiseuille_flow_sim()