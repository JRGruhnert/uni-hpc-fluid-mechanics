from abc import abstractmethod
import numpy as np
import lb

class Boundary():
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
 
    @abstractmethod
    def cache(self):
        """Called before stream and collide to cache the pre-stream values."""
        pass

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

    def cache(self, f):
        if (self.placement == 'top'):
            self.cached = f[:, :, 1]
        elif (self.placement == 'bottom'):
            self.cached = f[:, :, -2]
        elif (self.placement == 'left'):
            self.cached = f[:, 1, :]
        elif (self.placement == 'right'):
            self.cached = f[:, -2, :]
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))
    
    def apply(self, f):
        if (self.placement == 'top'):
            f[self.input_channels, -1, :] = self.cached[self.output_channels, -1, :]
        elif (self.placement == 'bottom'):
            f[self.input_channels, 0, :] = self.cached[self.output_channels, 0, None]
        elif (self.placement == 'left'):
            f[self.input_channels, :, 0] = self.cached[self.output_channels, :, 0]
        elif (self.placement == 'right'):
            f[self.input_channels, :, -1] = self.cached[self.output_channels, :, -1]
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))
    
    def update_velocity(self, velocities):
        if (self.placement == 'top'):
            velocities[:, :, 0] = 0.0
        elif (self.placement == 'bottom'):
            velocities[:, :, -1] = 0.0
        elif (self.placement == 'left'):
            velocities[:, 0,: ] = 0.0
        elif (self.placement == 'right'):
            velocities[:, -1,: ] = 0.0
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))
    


class MovingWall(Boundary):
    def __init__(self, velocity, placement='top', cs=1/np.sqrt(3)):
        super().__init__(placement)
        self.velocity = velocity
        self.cs = cs
    
    def calculate_wall_density(self, f):
        return lb.calculate_density(f[:,:,1])

    def update_f(self, f, value):
        f[self.input_channels,:,0] = value
    
    def cache(self, f):
        if (self.placement == 'top'):
            self.cached = f[:, :, 1]
        elif (self.placement == 'bottom'):
            self.cached = f[:, :, -2]
        elif (self.placement == 'left'):
            self.cached = f[:, 1, :]
        elif (self.placement == 'right'):
            self.cached = f[:, -2, :]
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))

    def apply(self, f, C, W):
        #coef = np.zeros((9, 100, 100))
        #value = 2 * W[self.output_channels] * self.calculate_wall_density(f) * (C[self.output_channels] @ self.velocity)/ self.cs ** 2
        #coef[self.output_channels, :, -1] = value[:, np.newaxis]
        #f[self.input_channels, :, -1] = self.cached[self.output_channels, :, -1] - value #coef[self.output_channels, :, -1]

        density = self.calculate_wall_density(f)
        """
        multiplier = 2 * (1/self.cs) ** 2
        temp = (C @ self.velocity)
        
        temp2 = density * W[:, None]
        
        momentum = multiplier * temp2 * temp[:, None]
        momentum = momentum[self.output_channels, :]
        temp3 = (self.cached[self.output_channels, :] - momentum)
        self.update_f(f, temp3 )
        """
        
        density = self.calculate_wall_density(f)
        multiplier = 2 * (1/self.cs) ** 2
        momentum = multiplier * (C @ self.velocity) * (W.T * density[:, None])
        momentum = momentum[:, self.output_channels].T
        self.update_f(f, (self.cache[:, self.output_channels].T - momentum).T)

    def update_velocity(self, velocities):
        if (self.placement == 'top'):
            velocities.T[:,0] = self.velocity
        elif (self.placement == 'bottom'):
            velocities[:, :, -1] = self.velocity
        elif (self.placement == 'left'):
            velocities[:, 0,: ] = self.velocity
        elif (self.placement == 'right'):
            velocities[:, -1,: ] = self.velocity
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))































class PeriodicWall(Boundary):
    def update_velocity(self, velocity_field):
        return super().update_velocity(velocity_field)
