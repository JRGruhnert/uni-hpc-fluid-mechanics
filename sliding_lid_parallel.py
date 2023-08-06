from lb import MpiWrapper
from plot import Plotter5

def sliding_lid_sim(nx: int = 300, ny: int = 300, omega: float = 0.3, steps: int = 20001):
    
    #assert(omega < 1.7)
    manager = MpiWrapper(nx, ny, 2, 2)

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

