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
                    left_address, right_address, 
                    bottom_address, top_address) -> None:
        # Send to and recieve from left
        recvbuf = self.f[:, 0, :].copy()
        comm.Sendrecv(sendbuf=self.f[:, 1, :].copy(), dest=left_address,
                  recvbuf=recvbuf, source=left_address)
        if left_address >= 0: self.f[:, 0, :] = recvbuf
        # Send to right
        recvbuf = self.f[:, -1, :].copy()
        comm.Sendrecv(sendbuf=self.f[:, -2, :].copy(), dest=right_address,
                  recvbuf=recvbuf, source=right_address)
        if right_address >= 0: self.f[:, -1, :] = recvbuf
        # Send to bottom
        recvbuf = self.f[:, :, 0].copy()
        comm.Sendrecv(sendbuf=self.f[:, :, 1].copy(), dest=bottom_address,
                  recvbuf=recvbuf, source=bottom_address)
        if bottom_address >= 0: self.f[:, :, 0] = recvbuf
        # Send to top
        recvbuf = self.f[:, :, -1].copy()
        comm.Sendrecv(sendbuf=self.f[:, :, -2].copy(), dest=top_address,
                  recvbuf=recvbuf, source=top_address)
        if top_address >= 0: self.f[:, :, -1] = recvbuf

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
    return np.dot(f.T, C).T / rho

def calculate_equilibrium(rho: np.ndarray, velocities: np.ndarray, dtype: np.dtype) -> np.ndarray:
    '''Calculate the equilibrium distribution function for a given density and velocity field'''
    test1 = np.dot(velocities.T, C.T).T
    test2 = np.sum(velocities**2, axis=0)
    return (W * (rho * (1 + 3 * test1 + 4.5 * test1**2 - 1.5 * test2)).T).T
        
    