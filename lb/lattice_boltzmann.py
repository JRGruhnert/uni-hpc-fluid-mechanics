import numpy as np
from lb.boundaries import Boundary, Periodic
from lb.helper import calculate_density, calculate_equilibrium, calculate_velocity_field
from lb.vars import C

class LatticeBoltzmann():
    def __init__(self, rho: np.ndarray, velocities: np.ndarray, omega: float, boundaries: list[Boundary] = []) -> None:
        self.rho = rho
        self.velocities = velocities
        self.omega = omega
        self.boundaries = boundaries
        self.f = calculate_equilibrium(self.rho, self.velocities)
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
        recvbuf = self.f[:, 0, :]
        comm.Sendrecv(sendbuf=self.f[:, 1, :], dest=left_address,
                  recvbuf=recvbuf, source=left_address)
        if left_address >= 0: self.f[:, 0, :] = recvbuf
        # Send to right
        recvbuf = self.f[:, -1, :]
        comm.Sendrecv(sendbuf=self.f[:, -2, :], dest=right_address,
                  recvbuf=recvbuf, source=right_address)
        if right_address >= 0: self.f[:, -1, :] = recvbuf
        # Send to bottom
        recvbuf = self.f[:, :, 0]
        comm.Sendrecv(sendbuf=self.f[:, :, 1], dest=bottom_address,
                  recvbuf=recvbuf, source=bottom_address)
        if bottom_address >= 0: self.f[:, :, 0] = recvbuf
        # Send to top
        recvbuf = self.f[:, :, -1]
        comm.Sendrecv(sendbuf=self.f[:, :, -2], dest=top_address,
                  recvbuf=recvbuf, source=top_address)
        if top_address >= 0: self.f[:, :, -1] = recvbuf

    def _pre_stream_boundaries(self) -> None:
        '''Bounce back particles from a wall'''
        for boundary in self.boundaries:
            if isinstance(boundary, Periodic):
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
        self.velocities = calculate_velocity_field(self.f, self.rho)
        self.f_eq = calculate_equilibrium(self.rho, self.velocities)
        self.f += self.omega * (self.f_eq - self.f)
