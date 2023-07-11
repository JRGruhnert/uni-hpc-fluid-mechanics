import numpy as np
from lb import LatticeBoltzmann, RigidWall, MovingWall
from plot import Plotter2

def couette_flow_sim(nx: int = 50, ny: int = 50, omega: float = 0.3, steps: int = 2000):
    
    rho = np.ones((nx, ny))
    velocities = np.zeros((2, nx, ny))
    wall_velocity = [0.0, 0.1]
    boundaries = [MovingWall(wall_velocity,"top"), RigidWall("bottom")]

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega, boundaries)
    plotter = Plotter2()

    for(step) in range(steps):
        latticeBoltzmann.tick()
        
        # plot shear wave every 200 steps 
        if((step % 200 == 0)):
            #rho, velocities = latticeBoltzmann.get_rho(), latticeBoltzmann.get_velocities()
            for boundary in latticeBoltzmann.boundaries:
                boundary.update_velocity(latticeBoltzmann.velocities)

            plotter.plot_cuette_flow(latticeBoltzmann.velocities, wall_velocity, step, nx, ny)
            print("Step: {}".format(step))

couette_flow_sim()
