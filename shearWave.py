import os
from matplotlib import pyplot as plt
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
        # update the lattice
        latticeBoltzmann.tick()  
        rho, velocities = latticeBoltzmann.output()

        # gather quantities for plotting
        plotter.gather_quantities(velocities, rho, p0, experiment)

        # plot shear wave every 200 steps 
        if(step % 200 == 0):
            plotter.plot_shear_wave(velocities, rho, p0, step, epsilon, nx, ny, experiment)

    # return the viscosity
    return plotter.return_viscosity(x, nx, epsilon, omega, steps, experiment)
    

# Experiments setup
omegas = np.arange(0.1, 2.01, 0.1)  
common_path = os.path.join('results', 'shear_wave_decay')
vis_path = os.path.join(common_path, 'viscosity')
os.makedirs(vis_path, exist_ok=True)
density_path = os.path.join(vis_path, f'density_viscosity.png')
velocity_path = os.path.join(vis_path, f'velocity_viscosity.png')

# Shear wave experiment 1 (density) 
density_simulated_viscosities = []
density_analytical_viscosities = []
for omega in omegas:
    simulated_viscosity, analytical_viscosity = shear_wave_sim(omega=omega, experiment='density')
    density_simulated_viscosities.append(simulated_viscosity)
    density_analytical_viscosities.append(analytical_viscosity)

# Shea wave experiment 2 (velocity)
velocity_simulated_viscosities = []
velocity_analytical_viscosities = []
for omega in omegas:
    simulated_viscosity, analytical_viscosity = shear_wave_sim(omega=omega, experiment='velocity')
    velocity_simulated_viscosities.append(simulated_viscosity)
    velocity_analytical_viscosities.append(analytical_viscosity)

# Plot density viscosity vs omega
plt.cla()
plt.scatter(omegas, np.log(density_simulated_viscosities), marker='x')
plt.scatter(omegas, np.log(density_analytical_viscosities), marker='x')
plt.xlabel('w')
plt.ylabel('Log(Viscosity)')
plt.legend(['Simulated', 'Analytical'])
plt.savefig(density_path, bbox_inches='tight', pad_inches=0)
    
# Plot velocity viscosity vs omega
plt.cla()
plt.scatter(omegas, np.log(velocity_simulated_viscosities), marker='x')
plt.scatter(omegas, np.log(velocity_analytical_viscosities), marker='x')
plt.xlabel('w')
plt.ylabel('Log(Viscosity)')
plt.legend(['Simulated', 'Analytical'])
plt.savefig(velocity_path, bbox_inches='tight', pad_inches=0)
   