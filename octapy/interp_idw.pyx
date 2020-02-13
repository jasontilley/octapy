import xarray as xr
from . cimport Data


def interp_idw(particle, grid, model, power=1.0):

#    cdef _Data data
    
    distances, indices = grid.tree.query([(particle.x,  particle.y)],
                                         k=model.leafsize)
    weights = 1. / distances ** power
    
    if model.dims == 2:
        nc_vars = ['u', 'v', 'temperature', 'salinity']
        part_vars = ['u', 'v', 'temp', 'sal']
        
    if model.dims == 3:
        nc_vars = ['u', 'v', 'w_velocity', 'temperature', 'salinity']
        part_vars = ['u', 'v', 'w', 'temp', 'sal']
    
    rootgrp = xr.open_dataset(particle.filepath)
    
    #data = Data()
#    data.u = rootgrp['u'].values
#    values = data.u.ravel()[indices][0]
#    value = (weights * values).sum() / weights.sum()
#    particle.u = value
    
#    for i, j in zip(nc_vars, part_vars):
#        setattr(data, j, rootgrp[i].values)
#        values = data.__getattribute__(j).ravel()[indices][0]
#        value = (weights * values).sum() / weights.sum()
#        setattr(particle, j, value)
        
    return(particle)
