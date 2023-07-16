from mpi4py import MPI


class WorkManager():
    def __init__(self, global_nx: int, global_ny: int, worker_grid_x: int, worker_grid_y: int ):
        self._workers = MPI.COMM_WORLD.Get_size()
        self._rank = MPI.COMM_WORLD.Get_rank()
        self._comm = MPI.COMM_WORLD.Create_cart(dims=[worker_grid_x, worker_grid_y], periods=[False, False], reorder=False)
        
        assert self._workers > worker_grid_x * worker_grid_y #max size

        self.left_src, self.left_dst = self._comm.Shift(0, -1)
        self.right_src, self.right_dst = self._comm.Shift(0, 1)
        self.bottom_src, self.bottom_dst = self._comm.Shift(1, -1)
        self.top_src, self.top_dst = self._comm.Shift(1, 1)
        
        worker_grid_x = global_nx//worker_grid_x
        worker_grid_y = global_ny//worker_grid_y
    
        # We need to take care that the total number of *local* grid points sums up to
        # nx. The right and topmost MPI processes are adjusted such that this is
        # fulfilled even if nx, ny is not divisible by the number of MPI processes.
        if self.right_dst < 0:
            # This is the rightmost MPI process
            local_nx = global_nx - local_nx*(worker_grid_x-1)
            without_ghosts_x = slice(0, local_nx)
        if self.right_dst >= 0:
            # Add ghost cell
            local_nx += 1
        if self.left_dst >= 0:
            # Add ghost cell
            local_nx += 1
            without_ghosts_x = slice(1, local_nx+1)
        if self.top_dst < 0:
            # This is the topmost MPI process
            local_ny = global_ny - local_ny*(worker_grid_y-1)
            without_ghosts_y = slice(0, local_ny)
        if self.top_dst >= 0:
            # Add ghost cell
            local_ny += 1
        if self.bottom_dst >= 0:
            # Add ghost cell
            local_ny += 1
            without_ghosts_y = slice(1, local_ny+1)

        #mpix, mpiy = self._comm.Get_coords(self._rank)
    
    def communicate(self, f_ikl) -> None:
        """
        Communicate boundary regions to ghost regions.

        Parameters
        ----------
        f_ikl : array
            Array containing the occupation numbers. Array is 3-dimensional, with
            the first dimension running from 0 to 8 and indicating channel. The
            next two dimensions are x- and y-position. This array is modified in
            place.
        """
        # Send to left
        recvbuf = f_ikl[:, -1, :].copy()
        self._comm.Sendrecv(f_ikl[:, 1, :].copy(), self.left_dst,
                    recvbuf=recvbuf, source=self.left_src)
        f_ikl[:, -1, :] = recvbuf
        # Send to right
        recvbuf = f_ikl[:, 0, :].copy()
        self._comm.Sendrecv(f_ikl[:, -2, :].copy(), self.right_dst,
                    recvbuf=recvbuf, source=self.right_src)
        f_ikl[:, 0, :] = recvbuf
        # Send to bottom
        recvbuf = f_ikl[:, :, -1].copy()
        self._comm.Sendrecv(f_ikl[:, :, 1].copy(), self.bottom_dst,
                    recvbuf=recvbuf, source=self.bottom_src)
        f_ikl[:, :, -1] = recvbuf
        # Send to top
        recvbuf = f_ikl[:, :, 0].copy()
        self._comm.Sendrecv(f_ikl[:, :, -2].copy(), self.top_dst,
                    recvbuf=recvbuf, source=self.top_src)
        f_ikl[:, :, 0] = recvbuf