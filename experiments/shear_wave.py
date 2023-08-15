import numpy as np
from lb.lattice_boltzmann import LatticeBoltzmann
from visualisation.plot import ShearWavePlotter

def shear_wave_sim(nx: int, ny: int, total_steps: int, plot_every: int, output_dir: str, 
                   omega: float, epsilon: float, p0: float, sub_experiment: str, viscosity: bool):

    rho = None
    velocities = None
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    #x, y = np.arange(nx), np.arange(ny)
    if(sub_experiment == "density"):
        # shear wave experiment 1 (density)
        rho = (p0 + epsilon * np.sin(2 * np.pi * x / nx)).T
        velocities = np.zeros((2, nx, ny))
    elif(sub_experiment == "velocity"):
         # shear wave experiment 2 (velocity)
        rho = np.ones((nx, ny))
        velocities = np.zeros((2, nx, ny))
        velocities[0, :, :] = (epsilon * np.sin(2 * np.pi * y / ny)).T
    else:
        print("Invalid experiment")
        return

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega)
    plotter = ShearWavePlotter(nx, ny, total_steps, output_dir, sub_experiment, p0, epsilon, omega)

    for(step) in range(total_steps):
        # update the lattice
        latticeBoltzmann.tick()  

        # get rho and velocities
        rho, velocities = latticeBoltzmann.rho, latticeBoltzmann.velocities
        
        # gather quantities for plotting
        if viscosity: plotter.gather_quantities(velocities, rho)

        # plot shear wave every 200 steps 
        if((step % plot_every == 0) and (not viscosity)):
            plotter.plot(velocities, rho, step)
            print("Step: {}".format(step))

    # return the viscosity
    return plotter.return_viscosity(total_steps) if viscosity else None