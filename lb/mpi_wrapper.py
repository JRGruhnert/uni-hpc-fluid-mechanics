import os
from mpi4py import MPI
import numpy as np
from numpy.lib.format import dtype_to_descr, magic
from lb.boundaries import RigidWall, TopMovingWall
from lb.lattice_boltzmann import LatticeBoltzmann


class MpiWrapper():
    def __init__(self, nx_global: int, ny_global: int, worker_dim_x: int, worker_dim_y: int):
        self.nx_worker_dim = worker_dim_x
        self.ny_worker_dim = worker_dim_y
        self._comm = MPI.COMM_WORLD
        self._workers = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

         # Check if the number of processes matches the grid dimensions
        if self._workers != self.nx_worker_dim * self.ny_worker_dim:
            raise ValueError("Number of processes does not match the grid dimensions.")
        
        self._cart_comm = self._comm.Create_cart((self.nx_worker_dim, self.ny_worker_dim), periods=(False, False))

        self.left_address, self.right_address = self._cart_comm.Shift(0, 1)
        self.bottom_address, self.top_address = self._cart_comm.Shift(1, 1)
        
        self.nx_local_buffered = nx_global//self.nx_worker_dim
        self.ny_local_buffered = ny_global//self.ny_worker_dim
        
        self.nx_local_without_buffer = slice(0,0) #initialize with dummy values
        self.ny_local_without_buffer = slice(0,0) #initialize with dummy values

        if self.right_address < 0:# This is the rightmost MPI process
            self.nx_local_buffered = nx_global - self.nx_local_buffered*(self.nx_worker_dim-1)
            self.nx_local_without_buffer = slice(1, self.nx_local_buffered + 1)
            self.nx_local_buffered += 1
        elif self.left_address < 0: # This is the leftmost MPI process
            self.nx_local_without_buffer = slice(0, self.nx_local_buffered)
            self.nx_local_buffered += 1
        else:
            self.nx_local_without_buffer = slice(1, self.nx_local_buffered + 2)
            self.nx_local_buffered += 2

        if self.top_address < 0:# This is the topmost MPI process
            self.ny_local_buffered = ny_global - self.ny_local_buffered*(self.ny_worker_dim-1)
            self.ny_local_without_buffer = slice(1, self.ny_local_buffered + 1)
            self.ny_local_buffered += 1
        elif self.bottom_address < 0: # This is the leftmost MPI process
            self.ny_local_without_buffer = slice(0, self.ny_local_buffered)
            self.ny_local_buffered += 1
        else:
            self.ny_local_without_buffer = slice(1, self.ny_local_buffered + 2)
            self.ny_local_buffered += 2

        mpix, mpiy = self._cart_comm.Get_coords(self._rank)
        rho = np.ones((self.nx_local_buffered, self.ny_local_buffered))
        velocities = np.zeros((2, self.nx_local_buffered, self.ny_local_buffered))
        wall_velocity = 0.05
        omega = 0.3
        boundaries = [TopMovingWall("top", wall_velocity), RigidWall("bottom"), RigidWall("left"), RigidWall("right")]
        self.lattice = LatticeBoltzmann(rho, velocities, omega, boundaries)
    
    def tick(self):
        self.lattice.tick()
        self.lattice.communicate(self._cart_comm, 
                                 self.left_address, self.right_address, 
                                 self.bottom_address, self.top_address)

    def save_mpiio(self, filename, index):
        PATH = "results"
        source='sliding_lid/mpi_raw'
        src_path = os.path.join(PATH, source)
        os.makedirs(src_path, exist_ok=True)
        filename = os.path.join(src_path, filename)
        
        local_velocities = self.lattice.velocities[index, self.nx_local_without_buffer, self.ny_local_without_buffer]
        nx_local = local_velocities.shape[0]
        ny_local = local_velocities.shape[1]

        magic_str = magic(1, 0)
        nx = np.empty_like(nx_local)
        ny = np.empty_like(ny_local)
        commx = self._cart_comm.Sub((True, False))
        commy = self._cart_comm.Sub((False, True))
        commx.Allreduce(np.asarray(nx_local), nx)
        commy.Allreduce(np.asarray(ny_local), ny)

        arr_dict_str = str({'descr': dtype_to_descr(local_velocities.dtype),
                            'fortran_order': False,
                            'shape': (nx.item(), ny.item())})

        while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
            arr_dict_str += ' '
        arr_dict_str += '\n'
        header_len = len(arr_dict_str) + len(magic_str) + 2

        x_offset = np.zeros_like(nx_local)
        commx.Exscan(np.asarray(ny * nx_local), x_offset)
        y_offset = np.zeros_like(ny_local)
        commy.Exscan(np.asarray(ny_local), y_offset)

        file = MPI.File.Open(self._cart_comm, filename, MPI.MODE_CREATE | MPI.MODE_WRONLY)
        if self._rank == 0:
            file.Write(magic_str)
            file.Write(np.int16(len(arr_dict_str)))
            file.Write(arr_dict_str.encode('latin-1'))
        mpitype = MPI._typedict[local_velocities.dtype.char]
        filetype = mpitype.Create_vector(nx_local, ny_local, ny)
        filetype.Commit()
        file.Set_view(header_len + (y_offset + x_offset) * mpitype.Get_size(),
                      filetype=filetype)
        file.Write_all(local_velocities.copy())
        filetype.Free()
        file.Close()


    def get_velocities(self):
        return self.lattice.velocities[:, self.nx_local_without_buffer, self.ny_local_without_buffer]
    
    def save_velocities(self, x_velocities_file, y_velocities_file):
        self.save_mpiio(x_velocities_file, 0)
        self.save_mpiio(y_velocities_file, 1)
        