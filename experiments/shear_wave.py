import numpy as np
from lb.lattice_boltzmann import LatticeBoltzmann
from visualisation.plot import ShearWavePlotter

def shear_wave_sim(nx: int, ny: int, total_steps: int, plot_every: int, output_dir: str, 
                   omega: float, epsilon: float, p0: float, sub_experiment: str, viscosity: bool):

    rho = None
    velocities = None
    #x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = np.arange(nx), np.arange(ny)
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
    
'''
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
for omega in omegas: # for viscosity plots
    simulated_viscosity, analytical_viscosity = shear_wave_sim(omega=omega, viscosity=True, experiment='density')
    density_simulated_viscosities.append(simulated_viscosity)
    density_analytical_viscosities.append(analytical_viscosity)
shear_wave_sim(experiment='density') # for shear wave plots

# Shea wave experiment 2 (velocity)
velocity_simulated_viscosities = []
velocity_analytical_viscosities = []
for omega in omegas: # for viscosity plots
    simulated_viscosity, analytical_viscosity = shear_wave_sim(omega=omega, viscosity=True, experiment='velocity')
    velocity_simulated_viscosities.append(simulated_viscosity)
    velocity_analytical_viscosities.append(analytical_viscosity)
shear_wave_sim(experiment='velocity') # for shear wave plots

# Plot density viscosity vs omega
plt.cla()
plt.scatter(omegas, np.log(density_simulated_viscosities), marker='x')
plt.scatter(omegas, np.log(density_analytical_viscosities), marker='x')
plt.xlabel('w')
plt.yscale('log')
plt.ylabel('Log(Viscosity)')
plt.legend(['Simulated', 'Analytical'])
plt.savefig(density_path, bbox_inches='tight', pad_inches=0)
    
# Plot velocity viscosity vs omega
plt.cla()
plt.scatter(omegas, np.log(velocity_simulated_viscosities), marker='x')
plt.scatter(omegas, np.log(velocity_analytical_viscosities), marker='x')
plt.xlabel('w')
plt.yscale('log')
plt.ylabel('Log(Viscosity)')
plt.legend(['Simulated', 'Analytical'])
plt.savefig(velocity_path, bbox_inches='tight', pad_inches=0)
   '''