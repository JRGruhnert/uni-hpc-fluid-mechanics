from abc import ABC, abstractmethod
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

from lb.vars import CS

class Plotter(ABC):
    def __init__(self, nx, ny, total_steps, experiment, path):
        self.fig, self.ax = plt.subplots()
        self.nx = nx
        self.ny = ny
        self.total_steps = total_steps
        self.experiment = experiment
        
        self.path = os.path.join(path, experiment)
        os.makedirs(self.path, exist_ok=True)
        
    @abstractmethod
    def plot(self):
        pass
    
        
class ShearWavePlotter(Plotter):
    def __init__(self, nx, ny, total_steps, path, sub_experiment, p0, epsilon, omega):
        super().__init__(nx, ny, total_steps, "shear_wave_decay/" + sub_experiment, path)
        self.quantities = []
        self.sub_experiment = sub_experiment
        self.p0 = p0
        self.epsilon = epsilon
        self.omega = omega

    def gather_quantities(self, velocities, rho):
        if self.sub_experiment == "density":
            self.quantities.append(np.max(np.abs(rho - self.p0)))
        else:
            self.quantities.append(np.max(np.abs(velocities[0, :, :])))

    def plot(self, velocities, rho, step):
        if self.sub_experiment == "density":
            self.ax.cla()
            self.ax.set_ylim([-self.epsilon + self.p0, self.epsilon + self.p0])
            self.ax.plot(np.arange(self.nx), rho[:, self.ny//2], '.')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('density')
            save_path = os.path.join(
                self.path, f'density_decay_{step}.png')
            self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            self.ax.cla()
            self.ax.set_ylim([-self.epsilon, self.epsilon])
            self.ax.plot(np.arange(self.ny), velocities[0, self.nx//2, :], '.')
            self.ax.set_xlabel('y')
            self.ax.set_ylabel('velocity')
            save_path = os.path.join(
                self.path, f'shear_wave_{step}.png')
            self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    def return_viscosity(self, steps):
        if self.sub_experiment == 'density':
            self.quantities = np.array(self.quantities)
            x = argrelextrema(self.quantities, np.greater)[0]
            self.quantities = self.quantities[x]
        else:
            x = np.arange(steps)

        coef = 2 * np.pi / self.nx
        simulated_viscosity = curve_fit(
            lambda t, visc: self.epsilon * np.exp(-visc * t * coef ** 2), xdata=x, ydata=self.quantities)[0][0]
        analytical_viscosity = (1/3) * ((1/self.omega) - 0.5)

        return simulated_viscosity, analytical_viscosity


class CouetteFlowPlotter(Plotter):
    def __init__(self, nx, ny, steps, plot_every, path , wall_velocity):
        super().__init__(nx, ny, steps, "couette_flow", path)
        self.padding_y = 0#0.5
        self.padding_x = 0#0.002
       
        self.y = np.arange(ny)
    
        self.analytical = (self.y) / (ny-1) * wall_velocity
        self.ax.set_xlim([-self.padding_x, wall_velocity + self.padding_x])
        self.ax.set_ylim([-self.padding_y, ny - 1 + self.padding_y])
        self.ax.set_ylabel('y position')
        self.ax.set_xlabel(f'Velocity $v_x$(x = {self.nx//2}, y)')
        # Seitenverhältnis einstellen
        ratio = abs((self.ax.get_xlim()[1] - self.ax.get_xlim()[0])/(self.ax.get_ylim()[0] - self.ax.get_ylim()[1])) * 0.75
        self.ax.set_aspect(ratio)

        # colorbar settings
        self.n_lines = steps // plot_every
        self.help_arr = []
        self.norm = mpl.colors.Normalize(vmin=0, vmax=steps)
        self.cmap = mpl.cm.ScalarMappable(norm=self.norm, cmap='cividis')
        self.cmap.set_array([])

    def plot(self, step, velocities):
        if step == 0 or step == (self.total_steps - 1):
            self.help_arr.append(step)

        self.ax.plot(velocities[0, self.nx//2, :], self.y, c=self.cmap.to_rgba(step + 1))#, color=COLORS[SIMULATION])

    def save(self, step):
        self.ax.plot(self.analytical, self.y, linestyle='dashed', c='darkred', label="Analytical Velocity")#, color=COLORS[ANALYTIC])
        self.ax.axhline(self.ny-1, linewidth=2, color='red', label='Moving Wall')
        self.ax.axhline(0.0, linewidth=2, color='black', label='Rigid Wall')
        save_path = os.path.join(self.path, f'couette_flow_{step}')
        self.ax.legend()  # add legend
        self.fig.colorbar(self.cmap, ticks=self.help_arr, label='Time Step', orientation='vertical')  # add colorbar
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


class PoiseuilleFlowPlotter(Plotter):
    def __init__(self, nx, ny, total_steps, path, omega, pressure_in, pressure_out):
        super().__init__(nx, ny, total_steps, "poiseuille_flow", path)
        self.omega = omega
        self.pressure_in = pressure_in
        self.pressure_out = pressure_out

    def plot(self, rho, velocities, step):
        y = np.arange(self.ny)
        self.ax.cla()
        viscosity = 1/3 * (1/self.omega - 0.5)
        dynamic_viscosity = rho[(self.nx-2)//2, :] * viscosity
        partial_derivative = (self.pressure_out - self.pressure_in) / (self.nx -2)
        analytical = -1 / (2 * dynamic_viscosity) * partial_derivative * y * (self.ny - 1 - y)
        
        #self.ax.set_xlim([0, np.max(analytical) + 0.002])
        self.ax.plot(analytical, y, linestyle='dashed', label="Analytical Velocity")
        self.ax.plot(velocities[0, (self.nx-2)//2, :], y, label="Simulated Velocity")
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('velocity')
        # plot walls
        self.ax.axhline(0, c='black', linewidth=3.5, label="Rigid Wall")  # indicate rigid bottom wall
        self.ax.axhline(self.ny-1, c='black', linewidth=3.5)  # indicate rigid bottom wall
        self.ax.legend()
        save_path = os.path.join(self.path, f'pousielle_flow_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


class SlidingLidPlotter(Plotter):
    def __init__(self, nx, ny, total_steps, path):
        super().__init__(nx, ny, total_steps, "sliding_lid/sequential", path)

        self.padding_y = 10
        self.padding_x = 10  
        self.x, self.y = np.meshgrid(np.arange(nx), np.arange(ny))
        

    def plot(self, velocities, step):
        self.ax.cla()
        speed = np.sqrt(velocities.T[self.y, self.x, 0] * velocities.T[self.y, self.x, 0] + velocities.T[self.y, self.x, 1]  * velocities.T[self.y, self.x, 1] )
        self.ax.streamplot(self.x, self.y, velocities.T[:, :, 0], velocities.T[:, :, 1], color=speed, cmap='RdBu_r')
       
        self.ax.set_xlim([0 - self.padding_x, self.nx + self.padding_x])
        self.ax.set_ylim([0 - self.padding_y, self.ny + self.padding_y])
        self.ax.set_ylabel('y-position')
        self.ax.set_xlabel('x-position')
        #self.ax.set_title(f'Step: {step}')
        save_path = os.path.join(self.path, f'{"sliding_lid"}_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

class Plotter5:
    def __init__(self, nx, ny, experiment, source, path):
        padding_y = 0.5
        padding_x = 0.002
        self.fig, self.ax = plt.subplots()
        self.experiment = experiment
        self.source = source
        self.ax.set_xlim([0, nx])
        self.ax.set_ylim([0, ny])
        self.x, self.y = np.meshgrid(np.arange(nx), np.arange(ny))
        self.path = os.path.join(path, experiment)
        os.makedirs(self.path, exist_ok=True)
        self.src_path = os.path.join(path, source)
        os.makedirs(self.src_path, exist_ok=True)

    def plot_sliding_lid_mpi(self, step, x_velocities_file, y_velocities_file):
        mpi_path_x = os.path.join(self.src_path, x_velocities_file)
        mpi_path_y = os.path.join(self.src_path, y_velocities_file)

        velocities_x, velocities_y = np.load(mpi_path_x), np.load(mpi_path_y)
        print("velocities x shape is: " + str(velocities_x.shape))
        #print("velocities x: " + str(velocities_x))
        print("velocities y shape is: " + str(velocities_y.shape))
        #print("velocities y: " + str(velocities_y))
        self.ax.cla()
        #self.ax.imshow(v, cmap='RdBu_r', vmin=0, interpolation='spline16')
        self.ax.streamplot(self.x, self.y, velocities_x.T, velocities_y.T, cmap='RdBu_r', density=0.8)
        #self.ax.legend(['Analytical'])
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('x')
        self.ax.set_title(f'Step: {step}')
        save_path = os.path.join(self.path, f'{"mpi_sliding_lid"}_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
