from lb.mpi_wrapper import MpiWrapper
from visualisation.plot import SlidingLidMpiPlotter

def sliding_lid_sim_mpi(nx: int, ny: int, total_steps: int, plot_every: int, output_dir: str, reynolds: int, wall_velocity: float):
    
    manager = MpiWrapper(nx, ny, 2, 2, reynolds, wall_velocity)

    plotter = SlidingLidMpiPlotter(nx, ny, total_steps, output_dir, "mpi_raw")

    for(step) in range(total_steps):
        manager.tick()
        # plot sliding lid every 1000 steps 
        if((step % plot_every == 0)):
            velocities_x_file = 'ux_{}.npy'.format(step)
            velocities_y_file = 'uy_{}.npy'.format(step)
            manager.save_velocities(output_dir, velocities_x_file, velocities_y_file)
            plotter.plot(step, velocities_x_file, velocities_y_file)
            print("Step: {}".format(step))


