from netCDF4 import Dataset

def interp_idw(particle, grid, model, float power=1.0):

    distances, indices = grid.tree.query([(particle.x,  particle.y)],
                                         k=model.leafsize)
    weights = (1. / distances[0] ** power)
    
    rootgrp = Dataset(particle.filepath)

    # read in the flattened data
    cdef float[:] u = rootgrp['u'][0].ravel()[indices][0]
    setattr(particle, 'u', sum(weights * u) / sum(weights))

    cdef float[:] v = rootgrp['v'][0].ravel()[indices][0]
    setattr(particle, 'v', sum(weights * v) / sum(weights))

    cdef float[:] temp = rootgrp['temperature'][0].ravel()[indices][0]
    setattr(particle, 'temp', sum(weights * temp) / sum(weights))

    cdef float[:] sal = rootgrp['salinity'][0].ravel()[indices][0]
    setattr(particle, 'sal', sum(weights * sal) / sum(weights))

    cdef float[:] w
    if model.dims == 3:
        w = rootgrp['w_velocity'][0].ravel()[indices][0]
        setattr(particle, 'w', sum(weights * w) / sum(weights))
        
    return(particle)
