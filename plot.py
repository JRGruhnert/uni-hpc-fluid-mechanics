import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

PATH="results"

class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.figs, self.axes = [self.fig, self.fig2], [self.ax, self.ax2]
        self.q = []
        self.path = os.path.join(PATH, 'shear_wave_decay')
        self.density_path = os.path.join(self.path, 'density')
        self.velocity_path = os.path.join(self.path, 'velocity')
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.density_path, exist_ok=True)
        os.makedirs(self.velocity_path, exist_ok=True)

    def init_shear_wave(self, experiment, nx, ny, epsilon, p0):
        pass

    def plot_shear_wave(self, velocities, rho, p0, step, epsilon, nx, ny, experiment):
        
        if experiment == "density":
            self.q.append(np.max(np.abs(rho - p0)))
        else:
            self.q.append(np.max(np.abs(velocities[1,:, :])))

        if(step % 200 != 0):
            return
        
        if experiment == "density":
            self.axes[0].cla()
            self.axes[0].set_ylim([-epsilon + p0, epsilon + p0])
            self.axes[0].plot(np.arange(nx), rho[ny//2,:])
            self.axes[0].set_xlabel('x')
            self.axes[0].set_ylabel('density')
            save_path = os.path.join(self.density_path, f'density_decay_{step}.png')
            self.figs[0].savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:  
            self.axes[0].cla()
            self.axes[0].set_ylim([-epsilon, epsilon])
            self.axes[0].plot(np.arange(ny), velocities[1,:,nx//2])
            self.axes[0].set_xlabel('y')
            self.axes[0].set_ylabel('velocity')
            save_path = os.path.join(self.velocity_path, f'shear_wave_{step}.png')
            self.figs[0].savefig(save_path, bbox_inches='tight', pad_inches=0)

     
    def return_viscosity(self, x, epsilon, omega, steps, experiment):
        if experiment == 'density':
            self.q = np.array(self.q)
            x = argrelextrema(self.q, np.greater)[0]
            self.q = self.q[x]
        else:
            x = np.arange(steps)
      
        coef = 2 * np.pi / 50
        simulated_viscosity = curve_fit(lambda t, visc: epsilon * np.exp(-visc * t * coef ** 2), xdata=x, ydata=self.q)[0][0]
        analytical_viscosity = (1/3) * ((1/omega) - 0.5)
        
        return simulated_viscosity, analytical_viscosity
    def close(self):
        plt.close(self.fig)
        plt.close(self.fig2)
