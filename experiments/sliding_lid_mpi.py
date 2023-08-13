from lb.mpi_wrapper import MpiWrapper
from visualisation.plot import Plotter5

def sliding_lid_sim_mpi(nx: int, ny: int, total_steps: int, plot_every: int, output_dir: str, reynolds: int, wall_velocity: float):
    
    manager = MpiWrapper(nx, ny, 2, 2)

    plotter = Plotter5(nx, ny, output_dir)

    for(step) in range(total_steps):
        manager.tick()
        # plot sliding lid every 1000 steps 
        if((step % plot_every == 0)):
            velocities_x_file = 'ux_{}.npy'.format(step)
            velocities_y_file = 'uy_{}.npy'.format(step)
            manager.save_velocities(velocities_x_file, velocities_y_file)
            plotter.plot_sliding_lid_mpi(step, velocities_x_file, velocities_y_file)
            print("Step: {}".format(step))


