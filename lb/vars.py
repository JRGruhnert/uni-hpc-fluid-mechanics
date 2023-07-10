import numpy as np


C = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], # X
              [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T  # Y
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

C.setflags(write=False)
W.setflags(write=False)

C_ALT = np.ascontiguousarray(
            np.array([[0, 0, 1, 0, -1, 1, 1, -1, -1], # y
                  [0, 1, 0, -1, 0, 1, -1, -1, 1]]).T) # x


W_ALT = np.ascontiguousarray(
            np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T,
            dtype=np.float32)

C_ALT.setflags(write=False)
W_ALT.setflags(write=False)