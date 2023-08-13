import argparse
import os

from matplotlib import pyplot as plt
import numpy as np
from experiments.couette_flow import couette_flow_sim
from experiments.poiseuille_flow import poiseuille_flow_sim

from experiments.shear_wave import shear_wave_sim
from experiments.sliding_lid import sliding_lid_sim
from experiments.sliding_lid_mpi import sliding_lid_sim_mpi


parser = argparse.ArgumentParser(
                    prog='Lattice Boltzmann Method',
                    description='Can run different LBM simulations',
                    epilog='Text at the bottom of help')
parser.add_argument('-s', '--simulation', type=str, 
                    choices=['shear_wave', 'shear_wave_viscosity', 'couette_flow', 'poiseuille_flow', 'sliding_lid', 'sliding_lid_mpi'], 
                    help='Simulation to run', required=True)

parser.add_argument('-nx', '--nx', type=int, help='Number of lattice points in x direction', default=300)
parser.add_argument('-ny', '--ny', type=int, help='Number of lattice points in y direction', default=300)
parser.add_argument('-t', '--total_steps', type=int, help='Total number of steps', required=True)
parser.add_argument('-p', '--plot_every', type=int, help='Plot every n steps', required=True)
parser.add_argument('-o', '--output_dir', type=str, help='Output directory', default='results')


args = parser.parse_args()
sub_parser = parser.add_subparsers(help='sub-command help')

dict_args = vars(args)
print(dict_args)
if dict_args['simulation'] == 'shear_wave':
    shear_wave_parser = sub_parser.add_parser('shear_wave', help='shear_wave help')
    shear_wave_parser.add_argument('-o', '--omega', type=float, help='Omega', required=True)
    shear_wave_parser.add_argument('-e', '--epsilon', type=float, help='Epsilon', required=True)
    shear_wave_parser.add_argument('-p', '--p0', type=float, help='P0', required=True)
    shear_wave_parser.add_argument('-e', '--experiment', type=str, choices=['density', 'velocity'], help='Experiment', required=True)
    args_shear_wave = shear_wave_parser.parse_args()
    shear_wave_sim(args[1], args[2], args[3], args[4], args_shear_wave[0], args_shear_wave[1], args_shear_wave[2], args_shear_wave[3], False)

if dict_args['simulation'] == 'shear_wave_viscosity':
    shear_wave_parser = sub_parser.add_parser('shear_wave', help='shear_wave help')
    shear_wave_parser.add_argument('-o1', '--omega1', type=float, help='Omega', required=True)
    shear_wave_parser.add_argument('-o2', '--omega2', type=float, help='Omega', required=True)
    shear_wave_parser.add_argument('-o3', '--omega3', type=float, help='Omega', required=True)
    shear_wave_parser.add_argument('-e', '--epsilon', type=float, help='Epsilon', required=True)
    shear_wave_parser.add_argument('-p', '--p0', type=float, help='P0', required=True)
    shear_wave_parser.add_argument('-e', '--experiment', type=str, choices=['density', 'velocity'], help='Experiment', required=True)
    args_shear_wave = shear_wave_parser.parse_args()

    sim_viscosities = []
    ana_viscosities = []
    omegas = [args_shear_wave[0], args_shear_wave[1], args_shear_wave[2]]
    for omega in omegas:
        sim_vis, ana_vis = shear_wave_sim(args[1], args[2], args[3], args[4], 
                                          omega, args_shear_wave[3], 
                                          args_shear_wave[4], args_shear_wave[5], 
                                          args_shear_wave[6], True)
        sim_viscosities.append(sim_vis)
        ana_viscosities.append(ana_vis)
    
    common_path = os.path.join('results', 'shear_wave_decay')
    vis_path = os.path.join(common_path, 'viscosity')
    os.makedirs(vis_path, exist_ok=True)
    path = os.path.join(vis_path, '{}_viscosity.png' .format(args_shear_wave[6]))

    # Plot density viscosity vs omega
    plt.cla()
    plt.scatter(omegas, np.log(sim_viscosities), marker='x')
    plt.scatter(omegas, np.log(ana_viscosities), marker='x')
    plt.xlabel('w')
    plt.yscale('log')
    plt.ylabel('Log(Viscosity)')
    plt.legend(['Simulated', 'Analytical'])
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
   
if dict_args['simulation'] == 'couette_flow':
    couette_flow_parser = sub_parser.add_parser('couette_flow', help='couette_flow help')
    couette_flow_parser.add_argument('-o', '--omega', type=float, help='Omega', required=True)
    couette_flow_parser.add_argument('-v', '--wall_velocity', type=float, help='Velocity of the top plate', required=True)
    args_couette_flow = couette_flow_parser.parse_args()
    couette_flow_sim(args[1], args[2], args[3], args[4], args_couette_flow[0], args_couette_flow[1])




if dict_args['simulation'] == 'poiseuille_flow':
    poiseuille_flow_parser = sub_parser.add_parser('poiseuille_flow', help='poiseuille_flow help')
    poiseuille_flow_parser.add_argument('-o', '--omega', type=float, help='Omega', required=True)
    poiseuille_flow_parser.add_argument('-pl', '--pressure_left', type=float, default=0.31, help='Pressure on the left', required=True)
    poiseuille_flow_parser.add_argument('-pr', '--pressure_right', type=float, default=0.3, help='Pressure on the right', required=True)
    args_poiseuille_flow = poiseuille_flow_parser.parse_args()
    poiseuille_flow_sim(args[1], args[2], args[3], args[4], args_poiseuille_flow[0], args_poiseuille_flow[1], args_poiseuille_flow[2])

if dict_args['simulation'] == 'sliding_lid':
    sliding_lid_parser = sub_parser.add_parser('sliding_lid', help='sliding_lid help')
    sliding_lid_parser.add_argument('-r', '--reynolds', type=int, help='Reynolds number', required=True)
    sliding_lid_parser.add_argument('-v', '--wall_velocity', type=float, help='Velocity of the top plate', required=True)
    args_sliding_lid = sliding_lid_parser.parse_args()
    sliding_lid_sim(args[1], args[2], args[3], args[4], args_sliding_lid[0], args_sliding_lid[1])

if dict_args['simulation'] == 'sliding_lid_mpi':
    sliding_lid_mpi_parser = sub_parser.add_parser('sliding_lid_mpi', help='sliding_lid_mpi help')
    sliding_lid_mpi_parser.add_argument('-r', '--reynolds', type=int, help='Reynolds number', required=True)
    sliding_lid_mpi_parser.add_argument('-v', '--wall_velocity', type=float, help='Velocity of the top plate', required=True)
    args_sliding_lid_mpi = sliding_lid_mpi_parser.parse_args()
    sliding_lid_sim_mpi(args[1], args[2], args[3], args[4], args_sliding_lid_mpi[0], args_sliding_lid_mpi[1])



