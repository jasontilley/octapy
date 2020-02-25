import numpy as np
#cimport numpy as np
from .netcdf cimport *
from .open_nc import open_nc
#from netCDF4 import Dataset


def interp_idw(particle, grid, int dims=2, int leafsize=3, float power=1.0):

    u, v, temp, sal = open_nc(particle.filepath)
    points = tuple(map(tuple, np.array([particle.x, particle.y]).T))
    distances, indices = grid.tree.query(points, k=leafsize, n_jobs=-1)
    weights = (1. / distances ** power).astype(np.float32)

    particle.u = (weights * np.array(u)[indices]).sum(axis=1) \
                  / weights.sum(axis=1)

    particle.v = (weights * np.array(v)[tuple([indices])]).sum(axis=1) \
                  / weights.sum(axis=1)

    particle.temp = (weights * np.array(temp)[tuple([indices])]).sum(axis=1) \
                     / weights.sum(axis=1)

    particle.sal = (weights * np.array(sal)[tuple([indices])]).sum(axis=1) \
                    / weights.sum(axis=1)

    return particle
    
    # Read in the flattened data. Beware, masked data is read in
    #cdef float[:,:] u = rootgrp['u'][0].ravel()[indices].data
    #particle.u = (weights * u).sum(axis=1) / weights.sum(axis=1)

    # cdef float[:,:] v = rootgrp['v'][0].ravel()[indices].data
    # particle.v = (weights * v).sum(axis=1) / weights.sum(axis=1)
    #
    # cdef float[:,:] temp = rootgrp['temperature'][0].ravel()[indices].data
    # particle.temp = ((weights * temp).sum(axis=1) / weights.sum(axis=1))
    #
    # cdef float[:,:] sal = rootgrp['salinity'][0].ravel()[indices].data
    # particle.sal = (weights * sal).sum(axis=1) / weights.sum(axis=1)
    #
    # cdef float[:,:] w
    # if dims == 3:
    #     w = rootgrp['w_velocity'][0].ravel()[indices].data
    #     particle.w = (weights * w).sum(axis=1) / weights.sum(axis=1)

