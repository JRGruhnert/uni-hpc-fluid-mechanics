import numpy as np
from lb.boundaries import Boundary, PortalWall
from lb.vars import C, W

class LatticeBoltzmann():
    def __init__(self, rho: np.ndarray, velocities: np.ndarray, omega: float, boundaries: list[Boundary] = [], dtype: np.dtype = np.float32) -> None:
        self.dtype = dtype
        self.rho = rho.astype(self.dtype)
        self.velocities = velocities.astype(self.dtype)
        self.omega = omega
        self.boundaries = boundaries
        self.f = calculate_equilibrium(self.rho, self.velocities, self.dtype)
        self.f_eq = self.f
        

    def tick(self) -> None:
        '''Executes one time step for the Lattice Boltzmann method'''
        self._pre_stream_boundaries()
        self._stream()
        self._after_stream_boundaries()
        self._collide()
    
    def communicate(self, comm, 
                    left_src, left_dst, 
                    right_src, right_dst, 
                    bottom_src, bottom_dst, 
                    top_src, top_dst) -> None:
        # Send to left
        recvbuf = self.f[:, -1, :].copy()
        comm.Sendrecv(self.f[:, 1, :].copy(), left_dst,
                  recvbuf=recvbuf, source=left_src)
        self.f[:, -1, :] = recvbuf
        # Send to right
        recvbuf = self.f[:, 0, :].copy()
        comm.Sendrecv(self.f[:, -2, :].copy(), right_dst,
                  recvbuf=recvbuf, source=right_src)
        self.f[:, 0, :] = recvbuf
        # Send to bottom
        recvbuf = self.f[:, :, -1].copy()
        comm.Sendrecv(self.f[:, :, 1].copy(), bottom_dst,
                  recvbuf=recvbuf, source=bottom_src)
        self.f[:, :, -1] = recvbuf
        # Send to top
        recvbuf = self.f[:, :, 0].copy()
        comm.Sendrecv(self.f[:, :, -2].copy(), top_dst,
                  recvbuf=recvbuf, source=top_src)
        self.f[:, :, 0] = recvbuf

    def _pre_stream_boundaries(self) -> None:
        '''Bounce back particles from a wall'''
        for boundary in self.boundaries:
            if isinstance(boundary, PortalWall):
                boundary.pre(self.f, self.f_eq, self.velocities)
            else:
                boundary.pre(self.f)

    def _stream(self) -> None:
        '''Stream particles'''
        for i in range(9):
            self.f[i] = np.roll(self.f[i], C[i], axis=(0, 1))
    
    def _after_stream_boundaries(self) -> None:
        '''Bounce back particles from a wall'''
        for boundary in self.boundaries:
            boundary.after(self.f)

    def _collide(self) -> None:
        '''Collide particles'''
        self.rho = calculate_density(self.f)
        self.velocities = calculate_velocity_field(self.f, self.rho, self.dtype)
        self.f_eq = calculate_equilibrium(self.rho, self.velocities, self.dtype)
        self.f += self.omega * (self.f_eq - self.f)

# Helper functions to calculate density, velocity field, equilibrium

def calculate_density(f: np.ndarray) -> np.ndarray:
    '''Calculate the density for a given distribution function'''
    return np.sum(f, axis=0)

def calculate_velocity_field(f: np.ndarray, rho: np.ndarray, dtype: np.dtype) -> np.ndarray:
    '''Calculate the velocity field for a given density and distribution function'''
    return np.dot(f.T, C.astype(dtype)).T / rho

def calculate_equilibrium(rho: np.ndarray, velocities: np.ndarray, dtype: np.dtype) -> np.ndarray:
    '''Calculate the equilibrium distribution function for a given density and velocity field'''
    test1 = np.dot(velocities.T, (C.T).astype(dtype)).T
    test2 = np.sum(velocities**2, axis=0)
    return (W.astype(dtype) * (rho * (1 + 3 * test1 + 4.5 * test1**2 - 1.5 * test2)).T).T
        
    