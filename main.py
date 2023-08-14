import argparse
import os
from matplotlib import pyplot as plt
import numpy as np

from experiments.couette_flow import couette_flow_sim
from experiments.poiseuille_flow import poiseuille_flow_sim
from experiments.shear_wave import shear_wave_sim
from experiments.sliding_lid import sliding_lid_sim
from experiments.sliding_lid_mpi import sliding_lid_sim_mpi

def run_shear_wave_sim(args):
    shear_wave_sim(args.nx, args.ny, args.total_steps, 
                   args.plot_every, args.output_dir, args.omega, 
                   args.epsilon, args.p0, args.experiment, False)
    

def run_shear_wave_viscosity_sim(args):
    sim_viscosities = []
    ana_viscosities = []
    for omega in args.omegas:
        sim_vis, ana_vis = shear_wave_sim(args.nx, args.ny, args.total_steps, 
                   args.plot_every, args.output_dir, omega, 
                   args.epsilon, args.p0, args.experiment, True)
        sim_viscosities.append(sim_vis)
        ana_viscosities.append(ana_vis)
    
    common_path = os.path.join('results', 'shear_wave_decay')
    vis_path = os.path.join(common_path, 'viscosity')
    os.makedirs(vis_path, exist_ok=True)
    path = os.path.join(vis_path, '{}_viscosity.png' .format(args.experiment))

    # Plot density viscosity vs omega
    plt.cla()
    plt.scatter(args.omegas, np.log(sim_viscosities), marker='x')
    plt.scatter(args.omegas, np.log(ana_viscosities), marker='x')
    plt.xlabel('w')
    plt.yscale('log')
    plt.ylabel('Log(Viscosity)')
    plt.legend(['Simulated', 'Analytical'])
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def run_couette_flow_sim(args):
    couette_flow_sim(args.nx, args.ny, args.total_steps, args.plot_every, 
                     args.output_dir, args.omega, args.wall_velocity)
    
def run_poiseuille_flow_sim(args):
    poiseuille_flow_sim(args.nx, args.ny, args.total_steps, args.plot_every, 
                        args.output_dir, args.omega, args.pressure_in, args.pressure_out)

def run_sliding_lid_sim(args):
    sliding_lid_sim(args.nx, args.ny, args.total_steps, args.plot_every, 
                    args.output_dir, args.reynolds, args.wall_velocity)

def run_sliding_lid_mpi_sim(args):
    sliding_lid_sim_mpi(args.nx, args.ny, args.total_steps, args.plot_every, 
                        args.output_dir, args.reynolds, args.wall_velocity)


parser = argparse.ArgumentParser(prog='Lattice Boltzmann Method', description='Can run different LBM simulations')
parser.add_argument('-nx', '--nx', type=int, help='Number of lattice points in x direction', default=100)
parser.add_argument('-ny', '--ny', type=int, help='Number of lattice points in y direction', default=50)
parser.add_argument('-ts', '--total_steps', type=int, help='Total number of steps', default=5000)
parser.add_argument('-ps', '--plot_every', type=int, help='Plot every n steps', default=500)
parser.add_argument('-out', '--output_dir', type=str, help='Output directory', default='results')

sub_parser = parser.add_subparsers(help='sub-command help', required=True)

shear_wave_parser = sub_parser.add_parser('shear_wave', help='shear_wave help')
shear_wave_parser.add_argument('-o', '--omega', type=float, help='Omega', default=1.0)
shear_wave_parser.add_argument('-ep', '--epsilon', type=float, help='Epsilon', default=0.01)
shear_wave_parser.add_argument('-p0', '--p0', type=float, help='P0', default=1.0)
shear_wave_parser.add_argument('-exp', '--experiment', type=str, choices=['density', 'velocity'], help='Experiment', required=True)
shear_wave_parser.set_defaults(func=run_shear_wave_sim)

shear_wave_viscosity_parser = sub_parser.add_parser('shear_wave_viscosity', help='shear_wave help')
shear_wave_viscosity_parser.add_argument('-os', '--omegas', nargs='+', type=float, help='Omega', default=[0.1, 2.01, 0.1])
shear_wave_viscosity_parser.add_argument('-eps', '--epsilon', type=float, help='Epsilon', default=0.1)
shear_wave_viscosity_parser.add_argument('-p0', '--p0', type=float, help='P0', default=1.0)
shear_wave_viscosity_parser.add_argument('-exp', '--experiment', type=str, choices=['density', 'velocity'], help='Experiment', required=True)
shear_wave_viscosity_parser.set_defaults(func=run_shear_wave_viscosity_sim)
 
couette_flow_parser = sub_parser.add_parser('couette_flow', help='couette_flow help')
couette_flow_parser.add_argument('-o', '--omega', type=float, help='Omega', default=0.3)
couette_flow_parser.add_argument('-wv', '--wall_velocity', type=float, help='Velocity of the top plate', default=0.1)
couette_flow_parser.set_defaults(func=run_couette_flow_sim)

poiseuille_flow_parser = sub_parser.add_parser('poiseuille_flow', help='poiseuille_flow help')
poiseuille_flow_parser.add_argument('-o', '--omega', type=float, help='Omega', default=0.3)
poiseuille_flow_parser.add_argument('-pi', '--pressure_in', type=float, default=1.005, help='Pressure on the left')
poiseuille_flow_parser.add_argument('-po', '--pressure_out', type=float, default=1.0, help='Pressure on the right')
poiseuille_flow_parser.set_defaults(func=run_poiseuille_flow_sim)
    
sliding_lid_parser = sub_parser.add_parser('sliding_lid', help='sliding_lid help')
sliding_lid_parser.add_argument('-rey', '--reynolds', type=int, help='Reynolds number', default=1000)
sliding_lid_parser.add_argument('-wv', '--wall_velocity', type=float, help='Velocity of the top plate', default=0.1)
sliding_lid_parser.set_defaults(func=run_sliding_lid_sim)

sliding_lid_mpi_parser = sub_parser.add_parser('sliding_lid_mpi', help='sliding_lid_mpi help')
sliding_lid_mpi_parser.add_argument('-rey', '--reynolds', type=int, help='Reynolds number', default=1000)
sliding_lid_mpi_parser.add_argument('-wv', '--wall_velocity', type=float, help='Velocity of the top plate', default=0.1)
sliding_lid_mpi_parser.set_defaults(func=run_sliding_lid_mpi_sim)

args = parser.parse_args()
args.func(args)