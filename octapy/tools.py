# octapy – Ocean Connectivity and Tracking Algorithms
# Copyright (C) 2020  Jason Tilley

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import octapy
from os.path import splitext
from netCDF4 import Dataset


def get_filepath(datetime64, model_name, submodel_name, data_dir):
    """ Get the filename for a given timestep

    Keyword arguments:
    datetime64 -- a numpy.datetime64 object for the data's time
    model_name -- the oceanographic model being used

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


def build_skill_release(drifter_file, model, period=pd.Timedelta('3 Days'),
                        data_freq=pd.Timedelta('60 minutes')):
    """Determine the model skill against a particular drifter

    Keyword arguments:
    drifter_file -- a .csv file of the drifter data with the columns 'datetime',
                    'lat', and 'lon'
    model -- an initialized octapy.tracking.Model object
    period -- a pandas Timedelta object representing the time period for which
              to calculate the skill
    data_freq -- a pandas Timedelta object representing the frequency of the
                 drifter data
    """

    # Read the release file
    release = pd.read_csv(model.release_file)
    release['start_time'] = pd.to_datetime(release['start_time'],
                                           format='%Y-%m-%dT%H:%M:%S')

    # Read the drifter data file
    drifter_data = pd.read_csv(drifter_file)

    # convert the dates to datetimes
    datetime = pd.to_datetime({'year': drifter_data['year'],
                               'month': drifter_data['month'],
                               'day': drifter_data['day'],
                               'hour': drifter_data['hour']})
    drifter_data['datetime'] = datetime

    # make sure you don't run the model outside the data range
    end_slice = int(period / data_freq)

    # set the stride for looping through date range
    stride = int(pd.Timedelta('1 Days') / data_freq)

    # for each particle in the release file
    for i in release.index:
        date_range = pd.date_range(release.iloc[i]['start_time'],
                                   release.iloc[i]['start_time']
                                   + pd.Timedelta(release.iloc[i]['days'], 'D'),
                                   freq=data_freq)
        date_range = date_range[:-end_slice:stride]
        # create a new dataframe to store the release data
        new_release = pd.DataFrame(columns=release.columns,
                                   index=range(0, len(date_range)))
        new_release['start_time'] = date_range.values.astype('datetime64[s]')
        new_release['particle_id'] = (release.iloc[i]['particle_id'].astype(str)
                                      + '_'
                                      + new_release.index.astype(str).str.zfill(2))
        new_release['num'] = release.iloc[i]['num']
        new_release['start_depth'] = release.iloc[i]['start_depth'].astype(str)
        new_release['days'] = period.days
        drifter_id = release.iloc[i]['particle_id']
        drifter_id_data = drifter_data[drifter_data['id'] == drifter_id]
        drifter_id_data = new_release.merge(drifter_id_data,
                                            left_on='start_time',
                                            right_on='datetime')
        new_release['start_lat'] = drifter_id_data['lat']
        new_release['start_lon'] = drifter_id_data['lon']
        new_release['start_time'] = date_range.values.astype('datetime64[s]').astype(str)
        new_release.to_csv('release_drifter_' + str(drifter_id) + '.csv',
                           float_format='%.4f', index=False)






