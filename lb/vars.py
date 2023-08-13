import numpy as np


C = np.ascontiguousarray(np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], # X
              [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T ) # Y
W = np.ascontiguousarray(np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]))

CS = 1/np.sqrt(3)  # speed of sound (in the context of this lattice simulation (D2Q9))

C.setflags(write=False)
W.setflags(write=False)
CS.setflags(write=False)