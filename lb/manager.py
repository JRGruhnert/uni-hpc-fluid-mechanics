from mpi4py import MPI
import numpy as np
from numpy.lib.format import dtype_to_descr, magic
from lb.boundaries import RigidWall, TopMovingWall

from lb.lattice_boltzmann import LatticeBoltzmann


class WorkManager():
    def __init__(self, global_nx: int, global_ny: int, worker_dim_x: int, worker_dim_y: int):
        self.worker_dim_x = worker_dim_x
        self.worker_dim_y = worker_dim_y
        self._comm = MPI.COMM_WORLD
        self._workers = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

         # Check if the number of processes matches the grid dimensions
        if self._workers != self.worker_dim_x * self.worker_dim_y:
            raise ValueError("Number of processes does not match the grid dimensions.")
        
        #print("Rank: {}".format(self._rank))
        #print("Workers: {}".format(self._workers))
        #print("Worker grid x: {}".format(self.worker_dim_x))
        #print("Worker grid y: {}".format(self.worker_dim_y))
        self._cart_comm = self._comm.Create_cart((self.worker_dim_x, self.worker_dim_y), periods=(False, False))
        
        #assert self._workers > worker_grid_x * worker_grid_y #max size

        self.left_src, self.left_dst = self._cart_comm.Shift(0, -1)
        self.right_src, self.right_dst = self._cart_comm.Shift(0, 1)
        self.bottom_src, self.bottom_dst = self._cart_comm.Shift(1, -1)
        self.top_src, self.top_dst = self._cart_comm.Shift(1, 1)
        
        self.local_nx = global_nx//self.worker_dim_x
        self.local_ny = global_ny//self.worker_dim_y
    
        # We need to take care that the total number of *local* grid points sums up to
        # nx. The right and topmost MPI processes are adjusted such that this is
        # fulfilled even if nx, ny is not divisible by the number of MPI processes.
        if self.right_dst < 0:
            # This is the rightmost MPI process
            self.local_nx = global_nx - self.local_nx*(self.worker_dim_x-1)
        self.without_ghosts_x = slice(0, self.local_nx)
        if self.right_dst >= 0:
            # Add ghost cell
            self.local_nx += 1
        if self.left_dst >= 0:
            # Add ghost cell
            self.local_nx += 1
            self.without_ghosts_x = slice(1, self.local_nx+1)
        if self.top_dst < 0:
            # This is the topmost MPI process
            self.local_ny = global_ny - self.local_ny*(self.worker_dim_y-1)
        self.without_ghosts_y = slice(0, self.local_ny)
        if self.top_dst >= 0:
            # Add ghost cell
            self.local_ny += 1
        if self.bottom_dst >= 0:
            # Add ghost cell
            self.local_ny += 1
            self.without_ghosts_y = slice(1, self.local_ny+1)

        #mpix, mpiy = self._comm.Get_coords(self._rank)
        rho = np.ones((self.local_nx, self.local_ny))
        velocities = np.zeros((2, self.local_nx, self.local_ny))
        wall_velocity = 0.05
        omega = 1.7
        boundaries = [RigidWall("bottom"), RigidWall("left"), RigidWall("right"), TopMovingWall("top", wall_velocity)]
        self.lattice = LatticeBoltzmann(rho, velocities, omega, boundaries)
    
    def tick(self):
        self.lattice.communicate(self._cart_comm, self.left_src, self.left_dst, self.right_src, self.right_dst, self.bottom_src, self.bottom_dst, self.top_src, self.top_dst)
        self.lattice.tick()
    

    def save_mpiio(self, filename, index):

        local_velocities = self.lattice.velocities[index, self.without_ghosts_x, self.without_ghosts_y]
        print("Local velocities shape on index: " + str(index) + " is: " + str(local_velocities.shape))
        magic_str = magic(1, 0)
        nx = np.empty_like(self.local_nx - 1)
        ny = np.empty_like(self.local_ny - 1)
        print("nx and ny on index: " + str(index) + " are: " + str(nx.shape) + " " + str(ny.shape))
        commx = self._cart_comm.Sub((True, False))
        commy = self._cart_comm.Sub((False, True))
        commx.Allreduce(np.asarray(self.local_nx - 1), nx)
        commy.Allreduce(np.asarray(self.local_ny - 1), ny)

        arr_dict_str = str({'descr': dtype_to_descr(local_velocities.dtype),
                            'fortran_order': False,
                            'shape': (nx.item(), ny.item())})

        while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
            arr_dict_str += ' '
        arr_dict_str += '\n'
        header_len = len(arr_dict_str) + len(magic_str) + 2

        x_offset = np.zeros_like(self.local_nx - 1)
        commx.Exscan(np.asarray(ny * self.local_nx - 1), x_offset)
        y_offset = np.zeros_like(self.local_ny - 1)
        commy.Exscan(np.asarray(self.local_ny - 1), y_offset)

        file = MPI.File.Open(self._cart_comm, filename, MPI.MODE_CREATE | MPI.MODE_WRONLY)
        if self._rank == 0:
            file.Write(magic_str)
            file.Write(np.int16(len(arr_dict_str)))
            file.Write(arr_dict_str.encode('latin-1'))
        mpitype = MPI._typedict[local_velocities.dtype.char]
        filetype = mpitype.Create_vector(self.local_nx - 1, self.local_ny - 1, ny)
        filetype.Commit()
        file.Set_view(header_len + (y_offset + x_offset) * mpitype.Get_size(),
                      filetype=filetype)
        file.Write_all(local_velocities.copy())
        filetype.Free()
        file.Close()


    def save_velocities(self, x_velocities_file, y_velocities_file):
        self.save_mpiio(x_velocities_file, 0)
        self.save_mpiio(y_velocities_file, 1)
        