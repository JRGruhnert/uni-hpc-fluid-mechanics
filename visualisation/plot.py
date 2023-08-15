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
        self.padding_y = epsilon * 1.1

    def gather_quantities(self, velocities, rho):
        if self.sub_experiment == "density":
            self.quantities.append(np.max(np.abs(rho - self.p0)))
        else:
            self.quantities.append(np.max(np.abs(velocities[0, :, :])))

    def calculate_analytical_sol(self, epsilon: float, viscosity: float, t: int, coordinates: np.ndarray, len_space: int):
        return epsilon * np.exp(- viscosity * (2.0 * np.pi / len_space) ** 2 * t) * np.sin(
            2.0 * np.pi / len_space * coordinates)

    def plot(self, velocities, rho, step):
        if self.sub_experiment == "density":
            self.ax.cla()
            self.ax.set_xlim([0, self.nx-1])
            self.ax.set_ylim([self.p0 - self.padding_y, self.p0 + self.padding_y])
            self.ax.plot(np.arange(self.nx), rho[:, self.ny//2])
            self.ax.set_xlabel('x-position')
            self.ax.set_ylabel(f'Density at position y={self.ny // 2}')
            #self.ax.set_title(f'Step: {step}')
            #self.ax.legend()
            save_path = os.path.join(
                self.path, f'density_decay_{step}.png')
            self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            self.ax.cla()
            self.ax.set_xlim([0, self.nx - 1])
            self.ax.set_ylim([-self.padding_y, self.padding_y])
            self.ax.plot(np.arange(self.ny), velocities[0, self.nx//2, :], label='Simulated')
            viscosity = CS ** 2 * (1.0 / self.omega - 0.5)
            self.ax.plot(np.arange(self.ny), self.calculate_analytical_sol(self.epsilon, viscosity, step, np.arange(self.ny), self.ny),
                        label="Analytical", linestyle='--')
            self.ax.set_xlabel('y-position')
            self.ax.set_ylabel(f'Velocity at position x={self.nx // 2}')
            if(step == 0):
                self.ax.legend()
            save_path = os.path.join(
                self.path, f'shear_wave_{step}.png')
            self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    def return_viscosity(self, steps):
        x = np.arange(steps)
        coef = 2 * np.pi / self.ny
        if self.sub_experiment == 'density':
            self.quantities = np.array(self.quantities)
            x = argrelextrema(self.quantities, np.greater)[0]
            self.quantities = self.quantities[x]
            coef = 2 * np.pi / self.nx
       
        sim_visc = curve_fit(
            lambda t, visc: self.epsilon * np.exp(-visc * t * coef ** 2), xdata=x, ydata=self.quantities)[0][0]
        ana_visc = (1/3) * ((1/self.omega) - 0.5)

        return sim_visc, ana_visc


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
        self.viscosity = 1/3 * (1/self.omega - 0.5)
        self.y = np.arange(self.ny)

    def plot(self, rho: np.ndarray, velocities, step):
        self.ax.cla()
        avg_density = 2 * rho[self.nx//2, :].mean() * self.viscosity # average density
        partial_derivative = CS**2 *(self.pressure_out - self.pressure_in) / self.nx # partial derivative of pressure
        analytical = -((1 / avg_density) * partial_derivative * self.y * (self.ny-1 - self.y)) # analytical velocity
        
        self.ax.set_xlim([0 - (np.max(analytical) * 1.05 - np.max(analytical)), np.max(analytical) * 1.1])
        self.ax.plot(velocities[0, self.nx//2, :], self.y, label="Simulated Velocity", c='red')
        self.ax.plot(analytical, self.y, linestyle='dashed', label="Analytical Velocity")
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('velocity')
        # plot walls
        self.ax.axhline(0, c='black', linewidth=2, label="Rigid Wall")  # indicate rigid bottom wall
        self.ax.axhline(self.ny-1, c='black', linewidth=2)  # indicate rigid bottom wall
        if(step == 0):
            self.ax.legend()
        save_path = os.path.join(self.path, f'pousielle_flow_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


class SlidingLidPlotter(Plotter):
    def __init__(self, nx, ny, total_steps, path, viscosity, wv):
        super().__init__(nx, ny, total_steps, "sliding_lid/sequential", path)

        self.padding_y = 5
        self.padding_x = 5 
        self.x, self.y = np.meshgrid(np.arange(nx), np.arange(ny))
        self.viscosity = viscosity
        self.wv = wv
        
    def plot(self, velocities, step):
        self.ax.cla()
        speed = np.sqrt(velocities.T[self.y, self.x, 0] ** 2  + velocities.T[self.y, self.x, 1] **2)
        self.ax.streamplot(self.x, self.y, velocities.T[:, :, 0], 
                           velocities.T[:, :, 1], color=speed, cmap='cool', density=0.8)
       
        self.ax.set_xlim([0 - self.padding_x, self.nx + self.padding_x])
        self.ax.set_ylim([0 - self.padding_y, self.ny + self.padding_y])
        self.ax.set_ylabel('y-position')
        self.ax.set_xlabel('x-position')
        #self.ax.set_title(f'Step: {step}')
        save_path = os.path.join(self.path, f'sl_{self.viscosity}_{self.wv}_{step}.png')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

class SlidingLidMpiPlotter(Plotter):
    def __init__(self, nx, ny, total_steps, path, source):
        super().__init__(nx, ny, total_steps, "sliding_lid/mpi", path)
        self.padding_y = 5
        self.padding_x = 5
        self.x, self.y = np.meshgrid(np.arange(nx), np.arange(ny))
        self.source = source

        self.src_path = os.path.join(self.path, source)
        os.makedirs(self.src_path, exist_ok=True)

    def plot(self, step, x_velocities_file, y_velocities_file):
        mpi_path_x = os.path.join(self.src_path, x_velocities_file)
        mpi_path_y = os.path.join(self.src_path, y_velocities_file)
        velocities_x, velocities_y = np.load(mpi_path_x), np.load(mpi_path_y)

        self.ax.cla()
        speed = np.sqrt(velocities_x.T[self.y, self.x] ** 2  + velocities_y.T[self.y, self.x] ** 2)
        self.ax.streamplot(self.x, self.y, velocities_x.T, velocities_y.T, color=speed, cmap='RdBu_r', density=0.8)
        #self.ax.legend(['Analytical'])

        self.ax.set_xlim([0 - self.padding_x, self.nx + self.padding_x])
        self.ax.set_ylim([0 - self.padding_y, self.ny + self.padding_y])
        self.ax.set_ylabel('y-position')
        self.ax.set_xlabel('x-position')
        #self.ax.set_title(f'Step: {step}')
        save_path = os.path.join(self.path, f'{"sl_mpi"}_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
