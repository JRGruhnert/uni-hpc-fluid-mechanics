from abc import abstractmethod, ABC
import numpy as np
import lb
from lb.vars import C_ALT, W

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
 
    def cache(self, f):
        if (self.placement == 'top'):
            self.f_cache = f[:, :, 1].copy()
        elif (self.placement == 'bottom'):
            self.f_cache = f[:, :, -2].copy()
        elif (self.placement == 'left'):
            self.f_cache = f[:, 1, :].copy()
        elif (self.placement == 'right'):
            self.f_cache = f[:, -2, :].copy()
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))

    @abstractmethod
    def apply(self):
        """Called after the stream and collide to apply boundary conditions."""
        pass

    @abstractmethod
    def update_velocity(self, f):
        pass


class RigidWall(Boundary):
    def __init__(self, placement='bottom'):
        super().__init__(placement)
    
    def apply(self, f):
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
    
    def update_velocity(self, velocities):
        if (self.placement == 'top'):
            velocities[:, :, 0] = 0.0
        elif (self.placement == 'bottom'):
            velocities[:, :, -1] = 0.0
        elif (self.placement == 'left'):
            velocities[:, 0, :] = 0.0
        elif (self.placement == 'right'):
            velocities[:, -1, :] = 0.0
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))
    


class MovingWall(Boundary):
    def __init__(self, placement, velocity, cs=1/np.sqrt(3)):
        self.velocity = velocity
        self.cs = cs
        super().__init__(placement)
    
    def calculate_wall_density(self, f):
        return lb.calculate_density(f[:, :, 1])

    def update_f(self, f, value):
        f[self.input_channels, :, 0] = value

    def apply(self, f):
        density = self.calculate_wall_density(f)
        multiplier = 2 * (1/self.cs) ** 2
        momentum = multiplier * (C_ALT @ self.velocity) * (W * density[: , None])
        momentum = momentum[:, self.output_channels]
        self.update_f(f, (self.f_cache.T[:, self.output_channels] - momentum).T)

    def update_velocity(self, velocities):
        if (self.placement == 'top'):
            velocities.T[0, :] = np.roll(self.velocity, 1)
        elif (self.placement == 'bottom'):
            velocities.T[-1, :] = np.roll(self.velocity, 1)
        elif (self.placement == 'left'):
            velocities.T[:, 0] = np.roll(self.velocity, 1)
        elif (self.placement == 'right'):
            velocities.T[:, -1] = np.roll(self.velocity, 1)
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))


class PortalWall(Boundary):
    def __init__(self, placement, n, pressure, cs=1/np.sqrt(3)):
        #only horizontal implementation
        if (placement == 'top' or placement == 'bottom'):
            raise ValueError("Invalid placement: {}".format(self.placement))
        super().__init__(placement)
        self.pressure = pressure / cs**2
        self.cs = cs
        self.n = n

    def cache(self, f, feq, velocities):
        if (self.placement == 'left'):
            temp_feq = lb.calculate_equilibrium(
                np.full(self.n, self.pressure),
                velocities[:, -2, :]).squeeze()
            self.f_cache = temp_feq + (f[:, -2, :] - feq[:, -2, :])
        else:
            temp_feq = lb.calculate_equilibrium(
                np.full(self.n, self.pressure),
                velocities[:, 1, :]).squeeze()
            self.f_cache = temp_feq + (f[:, 1, :] - feq[:, 1, :])

    def apply(self, f):
        if (self.placement == 'left'):
           f[self.input_channels, 0, :] = self.f_cache[self.input_channels, :]
        else:
           f[self.input_channels, -1, :] = self.f_cache[self.input_channels, :]


    def update_velocity(self, velocities):
        pass
