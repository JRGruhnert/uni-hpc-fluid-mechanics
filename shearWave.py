import numpy as np
from lb import LatticeBoltzmann
from plot import Plotter

def shear_wave_sim(experiment, nx: int = 50, ny: int = 50, omega: float = 1.0, epsilon: float = 0.01,
                          steps: int = 2000, p0: float = 1.0):
    
    # split velocity set into x and y components (unused right now)
    #ux = np.zeros((nx, ny))
    #uy = np.zeros((nx, ny))

    rho = None
    velocities = None
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))


    if(experiment == "density"):
       # shear wave experiment 1 (density)
        rho = p0 + epsilon * np.sin(2 * np.pi * x / nx)
        velocities = np.zeros((2, nx, ny))
    elif(experiment == "velocity"):
         # shear wave experiment 2 (velocity)
        rho = np.ones((nx, ny))
        velocities = np.zeros((2, nx, ny))
        velocities[1,:, :] = epsilon * np.sin(2 * np.pi * y / ny)
    else:
        print("Invalid experiment")
        return

    latticeBoltzmann = LatticeBoltzmann(rho, velocities, omega)
    plotter = Plotter()

    for(step) in range(steps):
        latticeBoltzmann.tick()   
        if(step % 200 == 0):
            rho, velocities = latticeBoltzmann.output()
            plotter.plot_shear_wave(velocities, rho, p0, step, epsilon, nx, ny, experiment)
    print("Finished simulation")


shear_wave_sim("density")