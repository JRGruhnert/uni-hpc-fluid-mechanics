from abc import abstractmethod, ABC
import numpy as np
from lb.helper import calculate_equilibrium
from lb.vars import C, CS, W

class Boundary(ABC):
    '''Abstract class for all kind boundaries'''
    def __init__(self, placement):
        '''Initializes the boundary and sets the input and output channels'''
        self.pre_called = False
        self.placement = placement
        if (placement == 'top'):
            self.input_channels = np.array([4, 7, 8]) 
            self.output_channels = np.array([2, 5, 6])
        elif (placement == 'bottom'):
            self.input_channels = np.array([2, 5, 6])
            self.output_channels =  np.array([4, 7, 8])
        elif (placement == 'left'):
            self.input_channels = np.array([3, 7, 6])
            self.output_channels = np.array([1, 5, 8])
        elif (placement == 'right'):
            self.input_channels = np.array([1, 8, 5])
            self.output_channels = np.array([3, 6, 7])
        else:
            raise ValueError("Invalid placement: {}".format(placement))
 
    def pre(self, f):
        """ Called before the streaming to cache boundary conditions 
            by saving the distribution function at the boundary."""
        if (self.placement == 'top'):
            self.f_cache = f[:, :, -1]
        elif (self.placement == 'bottom'):
            self.f_cache = f[:, :, 0]
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

    def _invalid_call_order(self, caller: str, pre_called: bool):
        '''Checks if the boundary was called in the wrong order.'''
        self.pre_called = pre_called 
        if (caller == 'pre' and self.pre_called):
            raise ValueError("Boundary pre-streaming called before after-streaming.")
        elif (caller == 'after' and not self.pre_called):
            raise ValueError("Boundary after-streaming called before pre-streaming.")




class RigidWall(Boundary):
    ''' Class for rigid walls where populations are bounced back at the boundary.'''
    def __init__(self, placement='bottom'):
        super().__init__(placement)
    
    def after(self, f):
        ''' Copys the cached pre-streaming distribution function to the 
            after-streaming distribution function at the boundary and flips the velocity component. '''
        if (self.placement == 'top'):
            f[self.input_channels, :, -1] = self.f_cache[self.output_channels]
        elif (self.placement == 'bottom'):
            f[self.input_channels, :, 0] = self.f_cache[self.output_channels]
        elif (self.placement == 'left'):
            f[self.input_channels, 0, :] = self.f_cache[self.output_channels]
        elif (self.placement == 'right'):
            f[self.input_channels, -1, :] = self.f_cache[self.output_channels]
        else:
            raise ValueError("Invalid placement: {}".format(self.placement))
    


class TopMovingWall(Boundary):
    ''' Class for moving walls where populations are bounced back at the boundary and a momentum is applied. '''
    def __init__(self, placement, wv):
        # only top implementation
        if (placement == 'bottom' or placement == 'left' or placement == 'right'):
            raise ValueError("Invalid placement: {}".format(self.placement))
        super().__init__(placement)
        self.velocity = [wv, 0.0] # velocity vector
    
    def after(self, f):
        ''' Copys the cached pre-streaming distribution function to the 
            after-streaming distribution function at the boundary and flips the velocity component. 
            And applies a momentum to the distribution function. '''
        rho = np.sum(f[:, :, 0], axis=0)
        factor = 2 * (1/CS) ** 2
        momentum = factor * (C @ self.velocity) * (W * rho[:, None])
        momentum = momentum[:, self.output_channels]
        f[self.input_channels, :, -1] = (self.f_cache[self.output_channels] - momentum.T)

class Periodic(Boundary):
    def __init__(self, placement, n, pressure):
        # only horizontal implementation
        if (placement == 'top' or placement == 'bottom'):
            raise ValueError("Invalid placement: {}".format(self.placement))
        super().__init__(placement)
        self.pressure = pressure # pressure at the boundary
        self.n = n # number of nodes in the y direction

    def pre(self, f, f_eq, velocities):
        ''' Applies the periodic boundary condition before streaming.'''
        if (self.placement == 'left'): #inlet
            temp_feq = calculate_equilibrium(
                np.full(self.n, self.pressure),
                velocities[:, -2, :]).squeeze()
            f[:, 0, :] = temp_feq + (f[:, -2, :] - f_eq[:, -2, :])
        else: #outlet
            temp_feq = calculate_equilibrium(
                np.full(self.n, self.pressure),
                velocities[:, 1, :]).squeeze()
            f[:, -1, :] = temp_feq + (f[:, 1, :] - f_eq[:, 1, :])

    def after(self, f):
        '''Nothing to do after streaming (periodic boundary condition)'''
        pass

