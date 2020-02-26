import numpy as np
from .netcdf cimport *
from .data import Data
from libcpp.vector cimport vector


def open_nc(filepath, dims=2):
    cdef nc_type var_type

    cdef int status = 0
    cdef char name[16]
    cdef int var_ndims
    cdef int var_dims[4]
    cdef int var_natts
    cdef size_t chunksizehint
    cdef int ncid
    nc__open(filepath.encode("utf-8"), 0, &chunksizehint, &ncid)
    cdef int u_id = ncvarid(ncid=ncid, name='u')
    cdef int v_id = ncvarid(ncid=ncid, name='v')
    cdef int w_id = ncvarid(ncid=ncid, name='w_velocity')
    cdef int temp_id = ncvarid(ncid=ncid, name='temperature')
    cdef int sal_id = ncvarid(ncid=ncid, name='salinity')
    cdef vector[size_t] start = np.intc(np.zeros(4))
    cdef vector[size_t] count = np.intc(np.array([1,1,385,541]))
    cdef float u[1*1*385*541]
    cdef float v[1*1*385*541]
    cdef float w[1*1*385*541]
    cdef float temp[1*1*385*541]
    cdef float sal[1*1*385*541]

    nc_get_vara_float(ncid, u_id, &start[0], &count[0], &u[0])
    nc_get_vara_float(ncid, v_id, &start[0], &count[0], &v[0])
    nc_get_vara_float(ncid, temp_id, &start[0], &count[0], &temp[0])
    nc_get_vara_float(ncid, sal_id, &start[0], &count[0], &sal[0])

    if dims == 3:
        nc_get_vara_float(ncid, w_id, &start[0], &count[0], &w[0])

    status = nc_close(ncid)
    if (status != 0):
        print('Error closing netcdf!')

    return Data(u, v, w, temp, sal)

