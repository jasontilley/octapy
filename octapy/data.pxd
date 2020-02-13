#import numpy as np
#cimport numpy as np

cdef extern from "data extern.cpp":
    pass

cdef extern from "data extern.h":

    cppclass _Data "Data":
        
        long datetime[1]
        double u, v, w, temp, sal
        Data() except +
        _Data(long*, double, double, double, double, double) except +
