import numpy as np

C = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], # X
              [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T  # Y
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

C.setflags(write=False)
W.setflags(write=False)


class LatticeBoltzmann():
    def __init__(self, rho, velocities, omega, boundaries=[]):
        self.rho = rho
        self.velocities = velocities
        self.omega = omega
        self.boundaries = boundaries
        self.f = calculate_equilibrium(self.rho, self.velocities)
        

    # Executes one time step for the Lattice Boltzmann method
    def tick(self):
        self.stream()
        self.collide()

    # Stream particles
    def stream(self):
        for i in range(9):
            self.f[i] = np.roll(self.f[i], C[i], axis=(0, 1))

    # Collide particles
    def collide(self):
        self.rho = calculate_density(self.f)
        self.velocities = calculate_velocity_field(self.f, self.rho)
        self.f += self.omega * (calculate_equilibrium(self.rho, self.velocities) - self.f)

    # Bounce back particles from a wall
    def bounce(self):
        for boundary in self.boundaries:
            pass
            #boundary.update_velocity(self.f)
    
    # Add a boundary to the simulation
    def add_boundary(self, boundary):
        self.boundaries.append(boundary)


    # Output for visualization
    def output(self):
        return self.rho, self.velocities




# Helper functions to calculate density, velocity field, equilibrium

def calculate_density(f):
    return np.sum(f, axis=0)

def calculate_velocity_field(f, rho):
    return np.dot(f.T, C).T / rho

def calculate_equilibrium(rho, velocities):
    test1 = np.dot(velocities.T, C.T).T
    test2 = np.sum(velocities**2, axis=0)
    return (W * (rho * (1 + 3 * test1 + 9/2 * test1**2 - 3/2 * test2)).T).T
        
    