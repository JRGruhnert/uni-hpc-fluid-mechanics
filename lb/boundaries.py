from abc import abstractmethod, ABC
import numpy as np
import lb
from lb.vars import C, W

class Boundary(ABC):
    def __init__(self, placement):
        self.placement = placement
        if (placement == 'top'):
            self.input_channels = np.array([2, 5, 6])
            self.output_channels = np.array([4, 7, 8])
        elif (placement == 'bottom'):
            self.input_channels = np.array([4, 7, 8])
            self.output_channels = np.array([2, 5, 6])
        elif (placement == 'left'):
            self.input_channels = np.array([1, 5, 8])
            self.output_channels = np.array([3, 7, 6])
        elif (placement == 'right'):
            self.input_channels = np.array([3, 6, 7])
            self.output_channels = np.array([1, 8, 5])
        else:
            raise ValueError("Invalid placement: {}".format(placement))
 
    def pre(self, f):
        """Called before the streaming to apply boundary conditions."""
        if (self.placement == 'top'):
            self.f_cache = f[:, :, 0]
        elif (self.placement == 'bottom'):
            self.f_cache = f[:, :, -1]
        elif (self.placement == 'left'):
            self.f_cache = f[:, 0, :]
        elif (self.placement == 'right'):
            self.f_cache = f[:, -1, :]
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))

    @abstractmethod
    def after(self):
        """Called after the streaming to apply boundary conditions."""
        pass


class RigidWall(Boundary):
    def __init__(self, placement='bottom'):
        super().__init__(placement)
    
    def after(self, f):
        if (self.placement == 'top'):
            f[self.input_channels, :, 0] = self.f_cache[self.output_channels, :]
        elif (self.placement == 'bottom'):
            f[self.input_channels, :, -1] = self.f_cache[self.output_channels, :]
        elif (self.placement == 'left'):
            f[self.input_channels, 0, :] = self.f_cache[self.output_channels, :]
        elif (self.placement == 'right'):
            f[self.input_channels, -1, :] = self.f_cache[self.output_channels, :]
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))
    


class TopMovingWall(Boundary):
    def __init__(self, placement, velocity, cs=1/np.sqrt(3)):
        #only horizontal implementation
        if (placement == 'bottom' or placement == 'left' or placement == 'right'):
            raise ValueError("Invalid placement: {}".format(self.placement))
        super().__init__(placement)
        self.velocity = [velocity, 0.0]
        self.cs = cs
    
    def after(self, f):
        rho = lb.calculate_density(f[:, :, 0])
        factor = 2 * (1/self.cs) ** 2
        momentum = factor * (C @ self.velocity) * (W * rho[:, None])
        momentum = momentum[:, self.output_channels]
        f[self.input_channels, :, 0] = (self.f_cache[self.output_channels] - momentum.T)

class PortalWall(Boundary):
    def __init__(self, placement, n, pressure, cs=1/np.sqrt(3)):
        #only horizontal implementation
        if (placement == 'top' or placement == 'bottom'):
            raise ValueError("Invalid placement: {}".format(self.placement))
        super().__init__(placement)
        self.pressure = pressure / cs**2
        self.cs = cs
        self.n = n

    def pre(self, f, f_eq, velocities):
        if (self.placement == 'left'):
            temp_feq = lb.calculate_equilibrium(
                np.full(self.n, self.pressure),
                velocities[:, -2, :]).squeeze()
            f[:, 0, :] = temp_feq + (f[:, -2, :] - f_eq[:, -2, :])
        else:
            temp_feq = lb.calculate_equilibrium(
                np.full(self.n, self.pressure),
                velocities[:, 1, :]).squeeze()
            f[:, -1, :] = temp_feq + (f[:, 1, :] - f_eq[:, 1, :])

    def after(self, f):
        pass

