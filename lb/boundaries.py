from abc import ABC, abstractmethod
import numpy as np

import lb

class Wall(ABC):
    def __init__(self):
        self.idxs = None

    @abstractmethod
    def update_velocity(self, velocity_field):
        pass


class RigidWall(Wall):
    def update_velocity(self, velocity_field):
        return super().update_velocity(velocity_field)

class MovingWall(Wall):
    def __init__(self, velocity, cs=1/np.sqrt(3)):
        super().__init__()
        self.velocity = velocity
        self.cs = cs
        
    def update_velocity(self, velocity_field):
        return super().update_velocity(velocity_field)
    