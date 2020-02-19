from libcpp.vector cimport vector
ctypedef struct _Data "Data":

    vector[long] datetime
    vector[double] u
    vector[double] v
    vector[double] w
    vector[double] temp
    vector[double] sal

