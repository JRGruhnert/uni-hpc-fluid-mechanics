# Helper functions to calculate density, velocity field, equilibrium

import numpy as np

from lb.vars import C, W


def calculate_density(f: np.ndarray) -> np.ndarray:
    '''Calculate the density for a given distribution function'''
    return np.sum(f, axis=0)

def calculate_velocity_field(f: np.ndarray, rho: np.ndarray) -> np.ndarray:
    '''Calculate the velocity field for a given density and distribution function'''
    return np.dot(f.T, C).T / rho

def calculate_equilibrium(rho: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    '''Calculate the equilibrium distribution function for a given density and velocity field'''
    test1 = np.dot(velocities.T, C.T).T
    test2 = np.sum(velocities ** 2, axis=0)
    return (W * (rho * (1 + 3 * test1 + 4.5 * test1**2 - 1.5 * test2)).T).T
        
    