from matplotlib import pyplot as plt
import numpy as np
from lb import WorkManager
from plot import Plotter4, Plotter5

def sliding_lid_sim(nx: int = 50, ny: int = 50, omega: float = 0.3, steps: int = 20001):
    
    #assert(omega < 1.7)
    manager = WorkManager(nx, ny, 2, 2)

    plotter = Plotter5(nx, ny)

    for(step) in range(steps):
        manager.tick()
        
        # plot sliding lid every 1000 steps 
        if((step % 1000 == 0)):
            velocities_x_file = 'ux_{}.npy'.format(step)
            velocities_y_file = 'uy_{}.npy'.format(step)
            manager.save_velocities(velocities_x_file, velocities_y_file)
            plotter.plot_sliding_lid_mpi(step, velocities_x_file, velocities_y_file)
            print("Step: {}".format(step))

sliding_lid_sim()

