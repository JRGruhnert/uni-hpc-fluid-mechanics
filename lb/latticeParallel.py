import numpy as np
from lb.manager import WorkManager 
from lb.boundaries import PortalWall, Boundary
from lb.vars import C, W

class LatticeBoltzmannParallel():
    def __init__(self, rho: np.ndarray, velocities: np.ndarray, omega: float, manager: WorkManager, boundaries: list[Boundary] = []) -> None:
        self.rho = rho
        self.velocities = velocities
        self.omega = omega
        self.manager = manager
        self.boundaries = boundaries
        self.f = calculate_equilibrium(self.rho, self.velocities)
        self.f_eq = self.f
        

    # Executes one time step for the Lattice Boltzmann method
    def tick(self) -> None:
        self._communicate()
        self._pre_stream_boundaries()
        self._stream()
        self._after_stream_boundaries()
        self._collide()

    # Stream particles
    def _stream(self) -> None:
        for i in range(9):
            self.f[i] = np.roll(self.f[i], C[i], axis=(0, 1))

    # Collide particles
    def _collide(self) -> None:
        self.rho = calculate_density(self.f)
        self.velocities = calculate_velocity_field(self.f, self.rho)
        self.f += self.omega * (self.f_eq - self.f)
        self.f_eq = calculate_equilibrium(self.rho, self.velocities)

    # Cache particles on boundary lattice points
    def _pre_stream_boundaries(self) -> None:
        for boundary in self.boundaries:
            if isinstance(boundary, PortalWall):
                boundary.pre(self.f, self.f_eq, self.velocities)
            else:
                boundary.pre(self.f)
    
     # Bounce back particles from boundary lattice points
    def _after_stream_boundaries(self) -> None:
        for boundary in self.boundaries:
            boundary.after(self.f)
    
    def _communicate(self) -> None:
        self.manager.communicate(self.f)

# Helper functions to calculate density, velocity field, equilibrium

def calculate_density(f: np.ndarray) -> np.ndarray:
    return np.sum(f, axis=0)

def calculate_velocity_field(f: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return np.dot(f.T, C).T / rho

def calculate_equilibrium(rho: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    test1 = np.dot(velocities.T, C.T).T
    test2 = np.sum(velocities**2, axis=0)
    return (W * (rho * (1 + 3 * test1 + 9/2 * test1**2 - 3/2 * test2)).T).T
        
    