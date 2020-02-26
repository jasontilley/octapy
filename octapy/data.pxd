
from libcpp.vector cimport vector
ctypedef struct _Data "Data":

    vector[float] u
    vector[float] v
    vector[float] w
    vector[float] temp
    vector[float] sal
