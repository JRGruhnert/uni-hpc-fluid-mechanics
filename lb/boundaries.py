from abc import abstractmethod, ABC
import numpy as np
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

    @abstractmethod
    def update_velocity(self, f):
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
    def __init__(self, placement, velocity, density, cs=1/np.sqrt(3)):
        self.velocity = velocity
        self.density = density[1]
        self.cs = cs
        super().__init__(placement)
    
    def _calculate_wall_density(self, f):
        return lb.calculate_density(f[:, :, 0])

    def _update_f(self, f, value):
        if (self.placement == 'top'):
            f[self.input_channels, :, 0] = value
        elif (self.placement == 'bottom'):
            f[self.input_channels, :, -1] = value
        elif (self.placement == 'left'):
            f[self.input_channels, 0, :] = value
        elif (self.placement == 'right'):
            f[self.input_channels, -1, :] = value
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))

    def after(self, f):
        density = self._calculate_wall_density(f)
        multiplier = 2 * (1/self.cs) ** 2
        momentum = multiplier * (C_ALT @ self.velocity) * (W * density[: , None])
        momentum = momentum[:, self.output_channels]
        self._update_f(f, (self.f_cache.T[:, self.output_channels] - momentum).T)

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


    def update_velocity(self, velocities):
        pass
