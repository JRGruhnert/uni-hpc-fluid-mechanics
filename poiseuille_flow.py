import numpy as np
from lb import LatticeBoltzmann, RigidWall, PressureWall
from plot import Plotter3

def poiseuille_flow_sim(nx: int = 50, ny: int = 50, pressure_left= 0.31, pressure_right= 0.3, omega: float = 0.3, steps: int = 2000):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    
    boundaries = [RigidWall("top"), RigidWall("bottom"), 
                  PressureWall(nx, pressure_left, placement="left"), PressureWall(nx, pressure_right, placement="right")]

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter3()

    for(step) in range(steps):
        latticeBoltzmann.tick()
        
        # plot shear wave every 200 steps 
        if((step % 200 == 0)):
            for boundary in latticeBoltzmann.boundaries:
                boundary.update_velocity(latticeBoltzmann.velocities)

            plotter.plot_poiseuille_flow(latticeBoltzmann.rho, latticeBoltzmann.velocities, omega, pressure_left, pressure_right, step, nx, ny)
            print("Step: {}".format(step))

poiseuille_flow_sim()