from abc import abstractmethod
import numpy as np

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
    def __init__(self, placement='top'):
        super().__init__(placement)

    def cache(self, f):
        pass
    
    def apply(self, f):
        if (self.placement == 'top'):
            f[self.input_channels, -1, :] = f[self.output_channels, -1, :]
        elif (self.placement == 'bottom'):
            f[self.input_channels, 0, :] = f[self.output_channels, 0, :]
        elif (self.placement == 'left'):
            f[self.input_channels, :, 0] = f[self.output_channels, 0, :]
        elif (self.placement == 'right'):
            f[self.input_channels, :, -1] = f[self.output_channels, -1, :]
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))
    
    def update_velocity(self, f):
        pass


class MovingWall(Boundary):
    def __init__(self, velocity, placement='top', cs=1/np.sqrt(3)):
        super().__init__(placement)
        self.velocity = velocity
        self.cs = cs
    
    def cache(self, f):
        pass

    def apply(self, f):
        pass

    def update_velocity(self, f):
        if (self.placement == 'top'):
            f[self.input_channels, -1, :] = f[self.output_channels, -1, :]
        elif (self.placement == 'bottom'):
            f[self.input_channels, 0, :] = f[self.output_channels, 0, :]
        elif (self.placement == 'left'):
            f[self.input_channels, :, 0] = f[self.output_channels, 0, :]
        elif (self.placement == 'right'):
            f[self.input_channels, :, -1] = f[self.output_channels, -1, :]
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))


class PeriodicWall(Boundary):
    def update_velocity(self, velocity_field):
        return super().update_velocity(velocity_field)
