import os
from matplotlib import pyplot as plt
import numpy as np
from lb import LatticeBoltzmann
from plot import Plotter
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

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
        #if(step % 200 == 0):
        rho, velocities = latticeBoltzmann.output()
        plotter.plot_shear_wave(velocities, rho, p0, step, epsilon, nx, ny, experiment)
   
    return plotter.return_viscosity(x, epsilon, omega, steps, experiment)
    

ws = np.arange(0.1, 2.01, 0.1)

       
        
simulated_viscosities = []
analytical_viscosities = []
for w in ws:
    simulated_viscosity, analytical_viscosity = shear_wave_sim(omega=w, experiment='velocity')
    simulated_viscosities.append(simulated_viscosity)
    analytical_viscosities.append(analytical_viscosity)
        
plt.cla()
plt.scatter(ws, np.log(simulated_viscosities), marker='x')
plt.scatter(ws, np.log(analytical_viscosities), marker='x')
plt.xlabel('w')
plt.ylabel('Log(Viscosity)')
plt.legend(['Simulated', 'Analytical'])
common_path = os.path.join('results', 'shear_decay')
os.makedirs(common_path, exist_ok=True)
path = os.path.join(common_path, f'viscosity_gjgj.png')
plt.savefig(path, bbox_inches='tight', pad_inches=0)
    