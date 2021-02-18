from libcpp.vector cimport vector
import numpy as np
from .netcdf cimport *


def get_data_at_index(filepath, vector[size_t] index,
                      vector[size_t] surf_index, dims=2):
    """
    :param filepath: path to the netCDF
    :param indices: the ravelled index to extract from the variables in the
    netCDF
    :param dims: the dimensions of the model
    :return: a Data object instance
    """

    nc_open(filepath.encode("utf-8"), 0, &ncid)
    cdef int u_id = ncvarid(ncid=ncid, name='u')
    cdef int v_id = ncvarid(ncid=ncid, name='v')
    cdef int w_id = ncvarid(ncid=ncid, name='w_velocity')
    cdef int temp_id = ncvarid(ncid=ncid, name='temperature')
    cdef int sal_id = ncvarid(ncid=ncid, name='salinity')
    cdef int ssh_id = ncvarid(ncid=ncid, name='ssh')

    cdef vector[size_t] count = np.intc([1, 1, 1, 1, 1])

    nc_get_vara_float(ncid, u_id, &index[0], &count[0], &u[0])
    nc_get_vara_float(ncid, v_id, &index[0], &count[0], &v[0])
    nc_get_vara_float(ncid, temp_id, &index[0], &count[0], &temp[0])
    nc_get_vara_float(ncid, sal_id, &index[0], &count[0], &sal[0])
    nc_get_vara_float(ncid, ssh_id, &surf_index[0], &count[0], &ssh[0])

    if dims == 3:
        nc_get_vara_float(ncid, w_id, &index[0], &count[0], &w[0])

    status = nc_close(ncid)
    if status != 0:
        print('Error closing netcdf!')

    return u, v, w, temp, sal, ssh

