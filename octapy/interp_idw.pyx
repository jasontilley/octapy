import numpy as np
import xarray as xr
from .data import Data


def interp_idw(particle, grid, model, power=1.0):

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

    # read in the flattened data
    # vector can't read in nan data
    datetime = rootgrp['MT'].values.ravel().astype(np.int_)
    u = rootgrp['u'].values.ravel()
    u[np.isnan(u)] = 0
    v = rootgrp['v'].values.ravel()
    v[np.isnan(v)] = 0
    w = rootgrp['w_velocity'].values.ravel()
    w[np.isnan(w)] = 0
    temp = rootgrp['temperature'].values.ravel()
    temp[np.isnan(temp)] = 0
    sal = rootgrp['salinity'].values.ravel()
    sal[np.isnan(sal)] = 0

    data = Data(datetime, u, v, w, temp, sal)
    
    for i, j in zip(nc_vars, part_vars):
        setattr(data, j, rootgrp[i].values.ravel())
        values = np.array(data.__getattribute__(j))[indices][0]
        value = (weights * values).sum() / weights.sum()
        setattr(particle, j, value)
        
    return(particle)
