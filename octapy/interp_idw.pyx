import numpy as np
#cimport numpy as np
from .netcdf cimport *
#from netCDF4 import Dataset


def interp_idw(particle, data, weights, int dims=2):

    particle.u = (weights * np.array(data.u)).sum(axis=1) \
                  / weights.sum(axis=1)
    particle.v = (weights * np.array(data.v)).sum(axis=1) \
                  / weights.sum(axis=1)
    particle.temp = (weights * np.array(data.temp)).sum(axis=1) \
                     / weights.sum(axis=1)
    particle.sal = (weights * np.array(data.sal)).sum(axis=1) \
                    / weights.sum(axis=1)

    if dims == 3:
        particle.w = (weights * np.array(data.w)).sum(axis=1) \
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

