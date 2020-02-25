# octapy – Ocean Connectivity and Tracking Algorithms
# Copyright (C) 2020  Jason Tilley

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from os.path import splitext
from netCDF4 import Dataset


def get_filepath(datetime64, model_name, submodel_name, data_dir):
    """ Get the filename for a given timestep

    Keyword arguments:
    datetime64 -- a numpy.datetime64 object for the data's time
    model -- the oceanographic model being used

    """
    datetime64s = datetime64 + np.timedelta64(30, 'm')
    datetime64s = np.datetime64(datetime64s, 'h')
    datetime64s = np.datetime64(datetime64s, 's')
    filepath = (data_dir + '/'
                + (''.join(filter(lambda x: x.isdigit(), str(datetime64)))
                   + '.' + model_name + '.' + submodel_name.replace('/', '.')
                   + '.nc'))
    return filepath


def get_extent(grid):
    extent = [grid.lons.min(), grid.lons.max(),
              grid.lats.min(), grid.lats.max()]
    return extent


def plot_csv_output(file_list, extent, step=2, plot_type='lines', colors=None):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')

    if colors is None:
        colors = np.repeat('blue', len(file_list))

    for csv_file, color in zip(file_list, colors):
        data = pd.read_csv(csv_file)
        lats = data['lat']
        lons = data['lon']

        if plot_type == 'lines':
            plt.plot(lons[0:-1:step], lats[0:-1:step], color=color,
                     linewidth=0.25)

        if plot_type == 'scatter':
            plt.scatter(lons[0:-1:step], lats[0:-1:step], color='blue', s=0.25)

        ax.set_extent(extent)

    plt.savefig(splitext(csv_file)[0] + '.png', dpi=600)
    plt.close()


def plot_netcdf_output(file_list, extent, step=2, plot_type='lines',
                       colors=None, drifter=None):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')

    if colors is None:
        colors = np.repeat('blue', len(file_list))

    for nc_file, color in zip(file_list, colors):
        rootgrp = Dataset(nc_file)
        lats = rootgrp['lat']
        lons = rootgrp['lon']

        if plot_type == 'lines':

            for i in range(0, len(lons)):
                plt.plot(lons[i, 0:-1:step], lats[i, 0:-1:step], color=color,
                         linewidth=0.25)

            if drifter is not None:
                data = pd.read_csv(drifter)
                lats = data['lat']
                lons = data['lon']
                plt.plot(lons, lats, color='orange', linewidth=0.25)

        if plot_type == 'scatter':

            for i in range(0, len(lats)):
                plt.scatter(lons[i, 0:-1:step], lats[i, 0:-1:step],
                            color='blue', s=0.25)

            if drifter is not None:
                data = pd.read_csv(drifter)
                lats = data['lat']
                lons = data['lon']
                plt.scatter(lons, lats, color='orange', s=0.25)

        ax.set_extent(extent)

    plt.savefig(splitext(nc_file)[0] + '.png', dpi=600)
    plt.close()
