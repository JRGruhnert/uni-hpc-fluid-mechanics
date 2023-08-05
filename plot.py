import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

PATH = "results"
COLORS = ["orange", "blue"]
ANALYTIC = 0
SIMULATION = 1
VELOCITY = 2
DENSITY = 3

plt.rcParams['mathtext.fontset'] = 'stix'  # The setting of math font
#plt.style.use('ggplot')


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.quantities = []

        self.path = os.path.join(PATH, 'shear_wave_decay')
        self.density_path = os.path.join(self.path, 'density')
        self.velocity_path = os.path.join(self.path, 'velocity')
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.density_path, exist_ok=True)
        os.makedirs(self.velocity_path, exist_ok=True)

    def gather_quantities(self, velocities, rho, p0, experiment):
        if experiment == "density":
            self.quantities.append(np.max(np.abs(rho - p0)))
        else:
            self.quantities.append(np.max(np.abs(velocities[0, :, :])))

    def plot_shear_wave(self, velocities, rho, p0, step, epsilon, nx, ny, experiment):
        if experiment == "density":
            self.ax.cla()
            self.ax.set_ylim([-epsilon + p0, epsilon + p0])
            self.ax.plot(np.arange(nx), rho[:, ny//2], '.', color=COLORS[SIMULATION])
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('density')
            save_path = os.path.join(
                self.density_path, f'density_decay_{step}.png')
            self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            self.ax.cla()
            self.ax.set_ylim([-epsilon, epsilon])
            self.ax.plot(np.arange(ny), velocities[0, nx//2, :], '.', color=COLORS[SIMULATION])
            self.ax.set_xlabel('y')
            self.ax.set_ylabel('velocity')
            save_path = os.path.join(
                self.velocity_path, f'shear_wave_{step}.png')
            self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    def return_viscosity(self, x, nx, epsilon, omega, steps, experiment):
        if experiment == 'density':
            self.quantities = np.array(self.quantities)
            x = argrelextrema(self.quantities, np.greater)[0]
            self.quantities = self.quantities[x]
        else:
            x = np.arange(steps)

        coef = 2 * np.pi / nx
        simulated_viscosity = curve_fit(
            lambda t, visc: epsilon * np.exp(-visc * t * coef ** 2), xdata=x, ydata=self.quantities)[0][0]
        analytical_viscosity = (1/3) * ((1/omega) - 0.5)

        return simulated_viscosity, analytical_viscosity


class Plotter2:
    def __init__(self, nx, ny, wall_velocity):
        padding_y = 0.5
        padding_x = 0.002
        self.lw_analytic = 2
        self.lw_simulated = 2
        self.fig, self.ax = plt.subplots()
        self.path = os.path.join(PATH, 'cuette flow')
        os.makedirs(self.path, exist_ok=True)
        self.nx = nx
        self.ny = ny
        self.y = np.arange(ny)
        self.analytical = (self.y) / (ny-1) * wall_velocity
        self.ax.set_xlim([0.0 - padding_x, wall_velocity + padding_x])
        self.ax.set_ylim([-padding_y, ny - 1 + padding_y])
        self.ax.axhline(self.ny-1, linewidth=2, color='red')
        self.ax.axhline(0.0, linewidth=2, color='black')
        self.ax.set_ylabel('y position (lattice units')
        self.ax.set_xlabel('Velocity u_x(x = 25, y)')
        self.ax.legend(['Moving Wall', 'Rigid Wall', 
                             'Analytical Velocity','Simulated Velocity'], loc='lower right')

    def plot_cuette_flow(self, velocities):
        self.ax.plot(velocities[0, self.nx//2, :], self.y, '.', markersize=self.lw_simulated)#, color=COLORS[SIMULATION])

    def save(self, step):
        self.ax.plot(self.analytical, self.y, lw=self.lw_analytic)#, color=COLORS[ANALYTIC])
        save_path = os.path.join(self.path, f'couette_flow_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


class Plotter3:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

        self.path = os.path.join(PATH, 'poiseuille flow')
        os.makedirs(self.path, exist_ok=True)

    def plot_poiseuille_flow(self, rho, velocities, omega, pressure_in, pressure_out, step, nx, ny):
        y = np.arange(ny)
        self.ax.cla()
        viscosity = 1/3 * (1/omega - 0.5)
        dynamic_viscosity = rho[nx//2, :] * viscosity
        partial_derivative = (pressure_out - pressure_in) / nx
        analytical = (-0.5 * partial_derivative * y *
                      (ny - 1 - y)) / dynamic_viscosity
        
        #shear_viscosity = (1 / omega - 0.5) / 3
        #delta = 2.0 / nx / shear_viscosity / 2.
        #y = np.linspace(0, ny, ny)
        #analytical = delta * y * (ny - y) / 3.

        # ax.set_xlim([0, np.max(analytical) + 0.001])
        self.ax.plot(analytical, y, COLORS[ANALYTIC])
        self.ax.plot(velocities[0, nx//2, :], y, '.', COLORS[SIMULATION])
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('velocity')
        self.ax.legend(['Analytical','Simulated'])
        save_path = os.path.join(self.path, f'pousielle_flow_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


class Plotter4:
    def __init__(self, nx, ny, experiment='sliding_lid/sequential'):
        padding_y = 0.5
        padding_x = 0.002
        self.fig, self.ax = plt.subplots()
        self.experiment = experiment
        self.ax.set_xlim([0, nx])
        self.ax.set_ylim([0, ny])
        self.x, self.y = np.meshgrid(np.arange(nx), np.arange(ny))
        self.path = os.path.join(PATH, experiment)
        os.makedirs(self.path, exist_ok=True)

    def plot_sliding_lid(self, velocities, step):
        self.ax.cla()
        #v = np.sqrt(velocities.T[:, :, 0]**2 + velocities.T[:, :, 1]**2)
        #self.ax.imshow(v, cmap='RdBu_r', vmin=0, interpolation='spline16')
        self.ax.streamplot(self.x, self.y, velocities.T[:, :, 0], velocities.T[:, :, 1], cmap='RdBu_r', density=0.8)
        #self.ax.legend(['Analytical'])
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('x')
        self.ax.set_title(f'Step: {step}')
        save_path = os.path.join(self.path, f'{self.experiment}_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

class Plotter5:
    def __init__(self, nx, ny, experiment='sliding_lid/parallel', source='sliding_lid/mpi_raw'):
        padding_y = 0.5
        padding_x = 0.002
        self.fig, self.ax = plt.subplots()
        self.experiment = experiment
        self.source = source
        self.ax.set_xlim([0, nx])
        self.ax.set_ylim([0, ny])
        self.x, self.y = np.meshgrid(np.arange(nx), np.arange(ny))
        self.path = os.path.join(PATH, experiment)
        os.makedirs(self.path, exist_ok=True)
        self.src_path = os.path.join(PATH, source)
        os.makedirs(self.src_path, exist_ok=True)

    def plot_sliding_lid_mpi(self, step, x_velocities_file, y_velocities_file):
        mpi_path_x = os.path.join(self.src_path, x_velocities_file)
        mpi_path_y = os.path.join(self.src_path, y_velocities_file)

        velocities_x, velocities_y = np.load(mpi_path_x), np.load(mpi_path_y)
        #velocities = np.stack([velocities_x, velocities_y], axis=-1)
        
        self.ax.cla()
        #v = np.sqrt(velocities.T[:, :, 0]**2 + velocities.T[:, :, 1]**2)
        #self.ax.imshow(v, cmap='RdBu_r', vmin=0, interpolation='spline16')
        self.ax.streamplot(self.x, self.y, velocities_x.T[:, :, 0], velocities_y.T[:, :, 1], cmap='RdBu_r', density=0.8)
        #self.ax.legend(['Analytical'])
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('x')
        self.ax.set_title(f'Step: {step}')
        save_path = os.path.join(self.path, f'{self.experiment}_{step}')
        self.fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
