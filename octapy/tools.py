#octapy – Ocean Connectivity and Tracking Algorithms
#Copyright (C) 2020  Jason Tilley

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from os.path import splitext
from numba import jit

def get_filepath(datetime64, model):
    ''' Get the filename for a given timestep
    
    Keyword arguments:
    datetime64 -- a numpy.datetime64 object for the data's time
    model -- the oceanographic model being used
    
    '''
    datetime64s = datetime64.astype('datetime64[m]') + np.timedelta64(30, 'm')
    datetime64s = datetime64s.astype('datetime64[h]')
    datetime64s = datetime64s.astype('datetime64[s]')
    filepath = (model.data_dir + '/'
                + (''.join(filter(lambda x: x.isdigit(), str(datetime64)))
                + '.' + model.model + '.' + model.submodel.replace('/', '.')
                + '.nc'))
    return(filepath)
    
    
def get_extent(grid):
    extent = [grid.lons.min(), grid.lons.max(),
              grid.lats.min(), grid.lats.max()]
    return(extent)


def plot_csv_output(file_list, extent, step=2):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    for csv_file in file_list:
        data = pd.read_csv(csv_file)
        lats = data['lat']
        lons = data['lon']
        plt.scatter(lons[0:-1:step], lats[0:-1:step], color='blue', s=0.25)
        ax.set_extent(extent)
        plt.savefig(splitext(csv_file)[0] + '.png', dpi=600)
        plt.close()
