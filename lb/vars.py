import numpy as np

# Lattice vectors (D2Q9)
C = np.ascontiguousarray(np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], # X
              [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T ) # Y

# Weights (D2Q9)
W = np.ascontiguousarray(np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]))

# speed of sound (D2Q9)
CS = 1/np.sqrt(3) 

C.setflags(write=False)
W.setflags(write=False)
CS.setflags(write=False)