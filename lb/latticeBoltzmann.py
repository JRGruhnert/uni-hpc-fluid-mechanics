import numpy as np

C = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], # X
              [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T  # Y
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

C.setflags(write=False)
W.setflags(write=False)


class LatticeBoltzmann():
    def __init__(self, density, velocity_field) -> None:
        self.density = density
        self.velocity_field = velocity_field
        self.f = self.calculate_equilibrium()
        #self.f = np.ones((9, 50, 50)) + 0.01 * np.random.randn(9, 50, 50)
        

    # Executes one time step for the Lattice Boltzmann method
    def tick(self):
        self.stream()
        self.collide()

    # Stream particles
    def stream(self):
        for i in range(9):
            self.f[i] = np.roll(self.f[i], C[i], axis=(0, 1))
            #self.f[:, :, i] = np.roll(self.f[:, :, i], CX, axis=1)
            #self.f[:, :, i] = np.roll(self.f[:, :, i], CY, axis=0)

    # Collide particles
    def collide(self, tau=0.6):
        self.density = self.calculate_density()
        self.velocity_field = self.calculate_velocity_field()
        self.f += 1/tau * (self.calculate_equilibrium() - self.f)

    # Output for visualization
    def output(self):
        return self.density, self.velocity_field

    # Helper functions to calculate density, velocity field, equilibrium

    def calculate_density(self):
        return np.sum(self.f, axis=0)

    def calculate_velocity_field(self):
        return np.dot(self.f.T, C).T / self.density

    def calculate_equilibrium(self):
        test1 = np.dot(self.velocity_field.T, C.T).T
        test2 = np.sum(self.velocity_field**2, axis=0)
        return (W * (self.density * (1 + 3 * test1 + 9/2 * test1**2 - 3/2 * test2)).T).T
        
        # for i in range(9):
        #    dot_product = np.dot(C[i], self.velocity_field)
        #    dot_product_squared = dot_product ** 2
        #    magnitude_squared = np.sum(self.velocity_field**2, axis=0)
        #
        #    term1 = W[i] * self.density
        #    term2 = 1 + 3 * dot_product + 9/2 * dot_product_squared - 3/2 * magnitude_squared
        #    self.feq[i] = term1 * term2
