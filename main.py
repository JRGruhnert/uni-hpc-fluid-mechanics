import argparse
from matplotlib import pyplot as plt
import numpy as np
import lb
from sim.simulation import Simulation


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--nx', type=int, default=50, help='Lattice Width')
    parser.add_argument('--ny', type=int, default=50, help='Lattice Height')
    parser.add_argument('--w', type=float, default=1.0, help='Omega')
    parser.add_argument('--eps', type=float, default=0.01, help='Magnitude')
    parser.add_argument('--n-steps', type=int, default=2000,
                        help='Number of simulation steps')
    parser.add_argument('--simulate', dest='simulate', action='store_true')
    parser.add_argument('--experiment-type', type=str, default='velocity',
                        choices=['velocity', 'density'])
    parser.add_argument('--p0', type=float, default=1.0, help='Density offset')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.set_defaults(simulate=False)
    args = parser.parse_args()

    simulation = Simulation(**vars(args))
    simulation.run(100)
