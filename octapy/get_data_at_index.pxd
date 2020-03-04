from .netcdf cimport *


cdef nc_type var_type
cdef int status = 0
cdef char name[32]
cdef int var_ndims
cdef int var_dims[4]
cdef int var_natts
cdef int ncid
cdef float u[1]
cdef float v[1]
cdef float w[1]
cdef float temp[1]
cdef float sal[1]
