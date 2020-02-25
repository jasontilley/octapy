cimport netcdf

cdef class Netcdf:
    cdef netcdf.Netcdf* _netcdf
    cdef int ncid
    cdef int varid
    cdef const long start[]
    cdef const long count[]
    cdef void *values



