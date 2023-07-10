from abc import abstractmethod, ABC
import numpy as np
import lb
from lb.vars import C, C_ALT, W, W_ALT

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
            self.input_channels = np.array([3, 7, 6])
            self.output_channels = np.array([1, 5, 8])
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
            velocities.T[0, :] = 0.0
        elif (self.placement == 'bottom'):
            velocities.T[-1, :] = 0.0
        elif (self.placement == 'left'):
            velocities.T[:, 0] = 0.0
        elif (self.placement == 'right'):
            velocities.T[:, -1] = 0.0
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))
    


class MovingWall(Boundary):
    def __init__(self, velocity, placement='top', cs=1/np.sqrt(3)):
        self.velocity = velocity
        self.cs = cs
        super().__init__(placement)
    
    def calculate_wall_density(self, f):
        return lb.calculate_density(f[:, :, 1])

    def update_f(self, f, value):
        f.T[0, :, self.input_channels] = value

    def apply(self, f):
        density = self.calculate_wall_density(f)
        multiplier = 2 * (1/self.cs) ** 2
        temp = (C @ self.velocity)
        temp2 = density * W[:, None]
        momentum = multiplier * temp2 * temp[:, None]
        momentum = momentum[self.output_channels, :]
        temp3 = (self.f_cache[self.output_channels, :] - momentum)
        self.update_f(f, temp3)
        

        
        #density = self.calculate_wall_density(f)
        #multiplier = 2 * (1/self.cs) ** 2
        #momentum = multiplier * (C_ALT @ self.velocity) * (W_ALT * density[: , None])
        #momentum = momentum[:, self.output_channels]
        #value = (self.f_cache.T[:, self.output_channels] - momentum).T
        #self.update_f(f, value )

    def update_velocity(self, velocities):
        if (self.placement == 'top'):
            velocities.T[0, :] = self.velocity
        elif (self.placement == 'bottom'):
            velocities.T[-1, :] = self.velocity
        elif (self.placement == 'left'):
            velocities.T[:, 0] = self.velocity
        elif (self.placement == 'right'):
            velocities.T[:, -1] = self.velocity
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))































class PeriodicWall(Boundary):
    def update_velocity(self, velocity_field):
        return super().update_velocity(velocity_field)
