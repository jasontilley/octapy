# octapy – Ocean Connectivity and Tracking Algorithms
# Copyright (C) 2020  Jason Tilley

import glob
from os import path, makedirs

import numpy as np
import urllib.request
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from netCDF4 import Dataset
from numba.experimental import jitclass
from numba import types
from scipy.interpolate import interpn, interp1d
from scipy.spatial.ckdtree import cKDTree

from .interp_idw import interp_idw
from .data import Data
from .get_data_at_index import get_data_at_index
from octapy.tools import get_filepath


# @jitclass([('release_file', types.unicode_type),
#            ('model', types.unicode_type),
#            ('submodel', types.unicode_type),
#            ('data_dir', types.unicode_type),
#            ('direction', types.unicode_type),
#            ('dims', types.uint8),
#            ('depth', types.optional(types.float32)),
#            ('extent', types.optional(types.ListType(types.float32))),
#            ('data_date_range', types.optional(types.NPDatetime('m')[:])),
#            ('data_freq', types.optional(types.NPTimedelta('m'))),
#            ('timestep', types.optional(types.NPTimedelta('m'))),
#            ('data_timestep', types.NPTimedelta('m')),
#            ('interp', types.unicode_type),
#            ('leafsize', types.int32),
#            ('vert_migration', types.boolean),
#            ('vert_array', types.optional(types.float32[24])),
#            ('output_file', types.optional(types.unicode_type)),
#            ('output_freq', types.NPTimedelta('m'))])
class Model:
    """ A particle tracking Model object

    Keyword arguments:
    release_file -- a file containing the particle coordinates and release times
    model -- name string of the ocean model used for input data (e.g, 'HYCOM')
    submodel -- name string of the submodel and/or experiment used for input
                data (e.g, 'GOMl0.04/expt_31.0')
    data_dir -- data directory path
    direction -- forcing direction through time. Must be  1 for forward or -1
                 for backward
    dims -- dimensionality of the model, must be 2 or 3
    diffusion -- if true, enables diffusion
    data_vars -- an numpy.ndarray of the variable names in the data file
    depth -- forcing depth if the model is 2-dimensional
    extent -- a list of extent coordinates as [minlat, maxlat, minlon, maxlon]
    data_date_range -- a numpy.ndarray object of numpy.datetime64 objects
                       containing the dates of the input data. WARNING: Must
                       match frequency of data
    timestep -- a numpy.timedelta64 object representing the timestep of the
                tracking model in minutes (e.g., np.timedelta64(60,'m'))
    data_freq -- a numpy.timedelta64 object representing the frequency of data
                 on the data server (e.g., np.timedelta64(60,'m'))
    data_timestep -- a numpy.timedelta64 object representing the timestep of the
                data to be downloaded in minutes (e.g., np.timedelta64(60,'m'))
    interp -- interpolation method. Supported are 'linear', 'nearest',
        'splinef2d', and 'idw'.
    leafsize -- number of nearest neighbors for some interpolation schemes
    vert_migration -- if True, the particle will undergo daily vertical
                      migrations which will override w velocities
    vert_array -- an array of length 24 representing the depth of a particle
                  over a 24-hour period when vert_mirgration is set to True
    output_file = base output file name
    output_freq = a numpy.timedelta64 object representing how often the
                  particle data will be output to the output file in minutes
                  (e.g., np.timedelta64(60,'m'))

    Returns:
    A Model object

    """

    def __init__(self, release_file=None, model=None, submodel=None,
                 data_dir='data', direction=1, dims=2, diffusion=False,
                 depth=None, extent=None, data_date_range=None,
                 timestep=np.timedelta64(60, 'm'),
                 data_freq=np.timedelta64(60, 'm'),
                 data_timestep=np.timedelta64(60, 'm'), interp='linear',
                 leafsize=9, vert_migration=False, vert_array=None,
                 output_file=None, output_freq=np.timedelta64(60, 'm')):
        self.release_file = release_file
        self.model = model
        self.submodel = submodel
        self.data_dir = data_dir
        self.direction = direction
        self.dims = dims
        self.diffusion = diffusion
        self.depth = depth
        self.extent = extent
        self.data_date_range = data_date_range
        self.timestep = timestep
        self.data_freq = data_freq
        self.data_timestep = data_timestep
        self.interp = interp
        self.leafsize = leafsize
        self.vert_migration = vert_migration
        self.vert_array = vert_array
        self.output_file = output_file
        self.output_freq = output_freq


# @jitclass([('src_crs', types.pyobject),
#            ('tgt_crs', types.pyobject),
#            ('file', types.unicode_type),
#            ('lats', types.float32[:]),
#            ('lons', types.float32[:]),
#            ('depths', types.float32[:]),
#            ('x', types.float32[:]),
#            ('y', types.float32[:]),
#            ('points', types.float32[:]),
#            ('tree', types.pyobject),
#            ('Dataset', types.pyobject)])

class Grid:
    """ A Grid object. You must have already downloaded the data into the data
    directory.

    Keyword arguments:
    model -- Model instance to which the Grid instance will belong

    Returns:
    A Grid object

    Attributes:
    src_crs -- a cartopy.crs projection object representing the data's source
               projection (e.g., cartopy.crs.LambertCylindrical())
    tgt_crs -- a cartopy.crs projection object representing the model's target
               projection
    file -- name of the file from which the grid was produced.
    lats -- latitudes of the grid
    lons -- longitudes of the grid
    depths -- depths of the grid
    x -- the x coordinates of the grid in meters
    y -- the y coordinates of the grid in meters
    points -- an array of grid coordinates in meters
    tree -- a scipy.spatial.cKDTree instance representing the grid

    """

    def __init__(self, model):

        self.src_crs = ccrs.LambertCylindrical()
        self.tgt_crs = ccrs.Mercator(central_longitude=-87.200012,
                                     min_latitude=18.091648,
                                     max_latitude=31.960648)
        self.file = glob.glob(model.data_dir + '/*')[0]
        data = Dataset(self.file)
        self.lats = data['Latitude'][:]
        self.lons = data['Longitude'][:]
        self.depths = data['Depth'][:]
        self.lons, self.lats = np.meshgrid(self.lons, self.lats)
        transformed = self.tgt_crs.transform_points(self.src_crs, self.lons,
                                                    self.lats)
        self.lons = self.lons[0]
        self.lats = self.lats.T[0]
        self.x = transformed[0, :, 0]
        self.y = transformed[:, 0, 1]

        x, y = np.meshgrid(self.x, self.y)
        self.points = np.array([self.x, self.y], dtype=object)
        # VisibleDeprecationWarning: Creating an ndarray from ragged nested
        # sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays
        # with different lengths or shapes) is deprecated. If you meant to do
        # this, you must specify 'dtype=object' when creating the ndarray
        coords = np.array([x.ravel(), y.ravel()]).T

        self.tree = cKDTree(coords, leafsize=model.leafsize)


# @jitclass([('lat', types.optional(types.float32[:])),
#            ('lon', types.optional(types.float32[:])),
#            ('depth', types.optional(types.float32[:])),
#            ('timestamp', types.optional(types.NPDatetime('s'))),
#            ('x', types.optional(types.float32[:])),
#            ('y', types.optional(types.float32[:])),
#            ('u', types.optional(types.float32[:])),
#            ('v', types.optional(types.float32[:])),
#            ('w', types.optional(types.float32[:])),
#            ('temp', types.optional(types.float32[:])),
#            ('sal', types.optional(types.float32[:])),
#            ('filepath', types.optional(types.unicode_type))])
class Particle:
    """ A Particle object

    Keyword arguments:
    lat -- Latitude of the particle
    lon -- Longitude of the particle
    depth -- Depth of the particle in meters
    timestamp -- a numpy.datetime64 instance representing the time of the
                 particle

    Returns:
    A Particle object

    Attributes:
    x -- the x coordinate in meters
    y -- the y coordinate in meters
    coord_tuple -- a tuple of the particle coordinates in meters
    u -- the u-velocity (eastward velocity) at the particle's location
    v -- the v-velocity (northward velocity) at the particle's location
    w -- the w-velocity (upward velocity) at the particle's location
    temp -- the temperature at the particle's location
    sal -- the salinity at the particle's location
    filepath -- the expected file path given the particles timestamp

    """

    def __init__(self, num=None, lat=None, lon=None, depth=None, timestamp=None,
                 x=None, y=None, u=None, v=None, w=None, temp=None, sal=None,
                 filepath=None):
        self.num = num
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.w = w
        self.temp = temp
        self.sal = sal
        self.filepath = filepath


def deepcopy(particle):
    particle_copy = Particle()
    particle_copy.lat = particle.lat
    particle_copy.lon = particle.lon
    particle_copy.depth = particle.depth
    particle_copy.timestamp = particle.timestamp
    particle_copy.x = particle.x
    particle_copy.y = particle.y
    particle_copy.u = particle.u
    particle_copy.v = particle.v
    particle_copy.w = particle.w
    particle_copy.temp = particle.temp
    particle_copy.sal = particle.sal
    particle_copy.filepath = particle.filepath

    return particle_copy


def transform(src_crs, tgt_crs, lon, lat):
    x, y = tgt_crs.transform_point(lon, lat, src_crs)
    return x, y


def download_hycom_data(model):

    if model.submodel == 'GOMl0.04/expt_20.1':
        base_prefix = 'http://ncss.hycom.org/thredds/ncss/grid/{0}/{1}'
    else:
        base_prefix = 'http://ncss.hycom.org/thredds/ncss/{0}/{1}/hrly'

    base_query = ('?var=salinity&var=temperature&var=u&var=v&var=w_velocity'
                  + '&time_start={0}&time_end={0}&accept=netcdf')

    if model.depth is not -1:
        depth_query = "&vertCoord=" + str(model.depth)
        base_query = base_query + depth_query

    if not path.exists(model.data_dir):
        makedirs(model.data_dir)

    start_time = np.datetime64(model.data_date_range[0])
    end_time = np.datetime64(model.data_date_range[-1])
    date_range = np.arange(start_time, end_time + model.data_timestep,
                           model.data_timestep)

    for date_time in date_range:
        curr_path = get_filepath(date_time, model.model, model.submodel,
                                 model.data_dir)

        if not path.exists(curr_path):
            prefix = base_prefix.format(model.submodel,
                                        str(date_time.astype(object).year))
            query = base_query.format(str(date_time))
            url = prefix + query
            urllib.request.urlretrieve(url, curr_path)

        else:
            print('File already exists!')


def get_physical(particle, grid, model):
    if path.isfile(particle.filepath):
        particle = interp3d(particle, grid, model)

    # else find the two surrounding files
    else:

        for i in range(1, 24):
            new_timestamp = particle.timestamp - np.timedelta64(i, 'h')
            file1 = get_filepath(new_timestamp, model.model, model.submodel,
                                 model.data_dir)
            if path.isfile(file1):
                particle1 = deepcopy(particle)
                particle1.timestamp = new_timestamp
                particle1.filepath = file1
                break

        for i in range(1, 24):
            new_timestamp = particle.timestamp + np.timedelta64(i, 'h')
            file2 = get_filepath(new_timestamp, model.model, model.submodel,
                                 model.data_dir)
            if path.isfile(file2):
                particle2 = deepcopy(particle)
                particle2.timestamp = new_timestamp
                particle2.filepath = file2
                break

        particle1 = interp3d(particle1, grid, model)
        particle2 = interp3d(particle2, grid, model)
        particle = interp_for_time(particle, particle1, particle2,
                                   dims=model.dims)

    return particle


def interp3d(particle, grid, model, power=1.0):
    # get the data here and feed to the interpolation functions
    leafsize = model.leafsize

    # get the vertical index of the nearest neighbor
    if model.dims == 3:
        dist_matrix = np.tile(grid.depths,
                              (len(particle.depth), 1)).T - particle.depth
        depth_indices = np.abs(dist_matrix).argmin(axis=0)

    # get the horizontal indices and weights
    points = tuple(map(tuple, np.array([particle.x, particle.y]).T))
    distances, indices = grid.tree.query(points, k=leafsize, n_jobs=-1)
    weights = (1. / distances ** power).astype(np.float32)

    data_shape = (1, 1, len(grid.lats), len(grid.lons))
    data = np.empty((len(indices), len(indices.T), 5))
    data.fill(np.nan)

    # for each particle
    for i in range(len(indices)):
        # get the four coordinate indices in the netcdf file
        start = np.intc(np.unravel_index(indices[i], data_shape))

        if model.dims == 3:
            start[1, :] = depth_indices[i]

        # get the data at the indices for each grid node
        for j, coord in zip(range(len(indices.T)), start.T):
            for k in reversed(range(coord[1] + 1)):
                data[i,j] = np.array(get_data_at_index(particle.filepath,
                                                       coord,
                                                       model.dims)).T[0]

                # Fix issue caused by depth being greater than bathymetry. If
                # depth is greater than bathymetry, set the depth to the next
                # shallower sigma level

                if not np.isnan(data).any():
                    break

                else:
                    coord[1] -= 1

        # interpolate the u, v, w, temp, and sal values
        if model.interp == 'idw':
            particle.u = (weights * np.array(data[:, :, 0])).sum(axis=1) \
                          / weights.sum(axis=1)
            particle.v = (weights * np.array(data[:, :, 1])).sum(axis=1) \
                          / weights.sum(axis=1)
            particle.w = (weights * np.array(data[:, :, 2])).sum(axis=1) \
                          / weights.sum(axis=1)
            particle.temp = (weights * np.array(data[:, :, 3])).sum(axis=1) \
                             / weights.sum(axis=1)
            particle.sal = (weights * np.array(data[:, :, 4])).sum(axis=1) \
                            / weights.sum(axis=1)

        # This is likely not working in 3D
        else:
            rootgrp = Dataset(particle.filepath)

            if model.dims == 2:
                dims_tuple = (particle.x, particle.y)

            if model.dims == 3:
                dims_tuple = (particle.x, particle.y, particle.depth)
                particle.w = interpn(grid.points, rootgrp['w_velocity'][0].T,
                                     dims_tuple, method=model.interp).item()

            particle.u = interpn(grid.points, rootgrp['u'][0].T,
                                 dims_tuple, method=model.interp).item()

            particle.v = interpn(grid.points, rootgrp['v'][0].T,
                                 dims_tuple, method=model.interp).item()

            particle.temp = interpn(grid.points, rootgrp['temperature'][0].T,
                                    dims_tuple, method=model.interp).item()

            particle.sal = interpn(grid.points, rootgrp['salinity'][0].T,
                                   dims_tuple, method=model.interp).item()

    return particle


# @jit(Particle.class_type.instance_type(Particle.class_type.instance_type,
#                                        Grid.class_type.instance_type,
#                                        Model.class_type.instance_type,
#                                        types.float32))
# def interp_idw(particle, grid, model, power=1.0):
#
#     distances, indices = grid.tree.query([(particle.x, particle.y)],
#                                          k=model.leafsize)
#     weights = (1. / distances[0] ** power)
#
#     rootgrp = Dataset(particle.filepath)
#
#     # read in the flattened data
#     u = rootgrp['u'][0].ravel()[indices][0]
#     setattr(particle, 'u', sum(weights * u) / sum(weights))
#
#     v = rootgrp['v'][0].ravel()[indices][0]
#     setattr(particle, 'v', sum(weights * v) / sum(weights))
#
#     temp = rootgrp['temperature'][0].ravel()[indices][0]
#     setattr(particle, 'temp', sum(weights * temp) / sum(weights))
#
#     sal = rootgrp['salinity'][0].ravel()[indices][0]
#     setattr(particle, 'sal', sum(weights * sal) / sum(weights))
#
#     if model.dims == 3:
#         w = rootgrp['w_velocity'][0].ravel()[indices][0]
#         setattr(particle, 'w', sum(weights * w) / sum(weights))
#
#     return (particle)


def interp_for_time(particle, particle1, particle2, dims=2):
    points = np.array([particle1.timestamp.astype(np.int_),
                       particle2.timestamp.astype(np.int_)])
    xi = particle.timestamp.astype(np.int_)

    f = interp1d(points, np.array([particle1.u, particle2.u]).T)
    particle.u = f(xi).astype(np.float32)

    f = interp1d(points, np.array([particle1.v, particle2.v]).T)
    particle.v = f(xi).astype(np.float32)

    f = interp1d(points, np.array([particle1.temp, particle2.temp]).T)
    particle.temp = f(xi).astype(np.float32)

    f = interp1d(points, np.array([particle1.sal, particle2.sal]).T)
    particle.sal = f(xi).astype(np.float32)

    if dims == 3:
        f = interp1d(points, np.array([particle1.w, particle2.w]).T)
        particle.w = f(xi).astype(np.float32)

    # u = np.interp(xi, points, np.array([particle1.u, particle2.u]))
    # particle.u = u
    # v = np.interp(xi, points, np.array([particle1.v, particle2.v]))
    # particle.v = v
    # temp = np.interp(xi, points, np.array([particle1.temp, particle2.temp]))
    # particle.temp = temp
    # sal = np.interp(xi, points, np.array([particle1.sal, particle2.sal]))
    # particle.sal = sal
    #
    # if dims == 3:
    #     u = np.interp(xi, points, np.array([particle1.u, particle2.u]))
    #     particle.u = u

    return particle


def force_particle(particle, grid, model):

    if model.diffusion == False:
        diffusion1 = 0
        diffusion2 = 0

    if model.diffusion == True:
        diffusion1 = np.random.normal(size=len(particle.x)) * .1
        diffusion2 = np.random.normal(size=len(particle.x)) * .1

    particle.x = (particle.x + model.timestep.item().seconds
                  * (particle.u + diffusion1) * model.direction)
    particle.y = (particle.y + model.timestep.item().seconds
                  * (particle.v + diffusion2) * model.direction)

    particle.timestamp = particle.timestamp + model.timestep * model.direction

    if model.dims == 3:

        if model.vert_migration == False:
            particle.depth = (particle.depth + model.timestep.item().seconds
                              * particle.w * model.direction)

        if model.vert_migration == True:
            particle.depth = np.repeat(model.vert_array[particle.timestamp.tolist().hour],
                                       particle.num)

    particle.filepath = get_filepath(particle.timestamp, model.model,
                                     model.submodel, model.data_dir)
    # update the particle's physical data
    particle = get_physical(particle, grid, model)

    return particle


def add_row_to_arr(arr, particle):
    row = np.array([particle.timestamp, particle.y, particle.x,
                    particle.depth, particle.u, particle.v, particle.w,
                    particle.temp, particle.sal])
    arr = np.vstack([arr, row])

    return arr


# def run_model(model, grid):
#     release = np.genfromtxt(model.release_file, delimiter=',', skip_header=1,
#                             dtype=[('particle_id', 'S8'),
#                                    ('start_lat', 'f8'),
#                                    ('start_lon', 'f8'),
#                                    ('start_depth', 'f8'),
#                                    ('start_time', '<M8[s]'),
#                                    ('days', 'f8')])
#
#     for i in range(0, len(np.atleast_1d(release))):
#         print('getting particle info for particle '
#               + np.atleast_1d(release['particle_id'])[i].astype(str))
#         trajectory = np.array(['timestamp', 'lat', 'lon', 'depth',
#                                'u', 'v', 'w', 'temp', 'sal'])
#         lat = np.atleast_1d(release['start_lat'])[i]
#         lon = np.atleast_1d(release['start_lon'])[i]
#         depth = np.atleast_1d(release['start_depth'])[i]
#
#         date_range = np.arange(np.atleast_1d(release['start_time'])[i],
#                                np.atleast_1d(release['start_time'])[i]
#                                + np.timedelta64(
#                                    int(np.atleast_1d(release['days'])[i]
#                                        * 24 * 60), 'm'),
#                                model.timestep)
#         particle = Particle(lat, lon, depth)
#         particle.x, particle.y = transform(grid.src_crs, grid.tgt_crs,
#                                            particle.lon, particle.lat)
#         for j in date_range:
#             particle.timestamp = j
#             particle.filepath = get_filepath(j, model.model,
#                                              model.submodel, model.data_dir)
#             # if j.tolist().hour == 0:
#             #     print('getting physical data for ' + str(j))
#             particle = get_physical(particle, grid, model)
#             trajectory = add_row_to_arr(trajectory, particle)
#             particle = force_particle(particle, grid, model)
#
#         trajectory = pd.DataFrame(trajectory[1::], columns=trajectory[0])
#         transformed = grid.src_crs.transform_points(grid.tgt_crs,
#                                                     np.array(trajectory['lon']),
#                                                     np.array(trajectory['lat'])).T
#         trajectory['lat'] = transformed[1]
#         trajectory['lon'] = transformed[0]
#         trajectory.to_csv(np.atleast_1d(release)['particle_id'][i].astype(str)
#                           + '_' + model.output_file, index=False)


def run_2d_model(model, grid):

    release = np.genfromtxt(model.release_file, delimiter=',', skip_header=0,
                            dtype=[('particle_id', 'S16'),
                                   ('num', 'i'),
                                   ('start_lat', 'f4'),
                                   ('start_lon', 'f4'),
                                   ('start_depth', 'f4'),
                                   ('start_time', '<M8[s]'),
                                   ('days', 'f4')])

    for i in range(0, len(np.atleast_1d(release))):
        print('getting particle info for particle '
              + np.atleast_1d(release['particle_id'])[i].astype(str))
        # add model.timestep to represent the true period
        #  model.timestep.astype(int)
        date_range = np.arange(np.atleast_1d(release['start_time'])[i],
                               np.atleast_1d(release['start_time'])[i]
                               + np.timedelta64(
                                   int(np.atleast_1d(release['days'])[i]
                                       * 24 * 60 + model.timestep.astype(int)),
                                   'm') * model.direction,
                               model.timestep * model.direction)
        # make array of dimensions (particle, time, data)
        num = np.atleast_1d(release['num'])[i]
        trajectory = np.empty((num, len(date_range), 7))
        lat = np.repeat(np.atleast_1d(release['start_lat'])[i], num)
        lon = np.repeat(np.atleast_1d(release['start_lon'])[i], num)
        depth = np.repeat(np.atleast_1d(release['start_depth'])[i], num)

        particle = Particle(num, lat, lon, depth)
        transformed = grid.tgt_crs.transform_points(grid.src_crs,
                                                    lon,
                                                    lat).T
        particle.x, particle.y = transformed[:2].astype(np.float32)
        particle.filepath = get_filepath(date_range[0], model.model,
                                         model.submodel, model.data_dir)
        particle.timestamp = date_range[0]
        particle = get_physical(particle, grid, model)
        for j in range(len(date_range)):
            #particle.timestamp = date_range[j]
            particle.filepath = get_filepath(particle.timestamp, model.model,
                                             model.submodel, model.data_dir)
            # if j.tolist().hour == 0:
            #    print('getting physical data for ' + str(j))
            particle_arr = np.array([particle.x, particle.y, particle.depth,
                                     particle.u, particle.v, particle.temp,
                                     particle.sal]).T
            trajectory[:, j, :] = particle_arr
            particle = force_particle(particle, grid, model)

        transformed = grid.src_crs.transform_points(grid.tgt_crs,
                                                    trajectory[..., 0],
                                                    trajectory[..., 1])
        lons = transformed[..., 0]
        lats = transformed[..., 1]

        # write the data to netcdf
        nc_data = xr.Dataset({'lat': (['release', 'time'], lats),
                              'lon': (['release', 'time'], lons),
                              'depth': (['release', 'time'], trajectory[..., 2]),
                              'u': (['release', 'time'], trajectory[..., 3]),
                              'v': (['release', 'time'], trajectory[..., 4]),
                              'temp': (['release', 'time'], trajectory[..., 5]),
                              'sal': (['release', 'time'], trajectory[..., 6])},
                             coords={'release': np.arange(0, num),
                                     'time': date_range})

        nc_data.to_netcdf('output/'
                          + np.atleast_1d(release['particle_id'])[i].astype(str)
                          + '_' + model.output_file)
        nc_data.close()

def run_3d_model(model, grid):

    release = np.genfromtxt(model.release_file, delimiter=',', skip_header=0,
                            dtype=[('particle_id', 'S16'),
                                   ('num', 'i'),
                                   ('start_lat', 'f4'),
                                   ('start_lon', 'f4'),
                                   ('start_depth', 'f4'),
                                   ('start_time', '<M8[s]'),
                                   ('days', 'f4')])

    for i in range(0, len(np.atleast_1d(release))):
        print('getting particle info for particle '
              + np.atleast_1d(release['particle_id'])[i].astype(str))
        # add model.timestep to represent the true period
        #  model.timestep.astype(int)
        date_range = np.arange(np.atleast_1d(release['start_time'])[i],
                               np.atleast_1d(release['start_time'])[i]
                               + np.timedelta64(
                                   int(np.atleast_1d(release['days'])[i]
                                       * 24 * 60 + model.timestep.astype(int)),
                                   'm') * model.direction,
                               model.timestep * model.direction)
        # make array of dimensions (particle, time, data)
        num = np.atleast_1d(release['num'])[i]
        trajectory = np.empty((num, len(date_range), 8)) ###
        lat = np.repeat(np.atleast_1d(release['start_lat'])[i], num)
        lon = np.repeat(np.atleast_1d(release['start_lon'])[i], num)

        if model.vert_migration == False:
            depth = np.repeat(np.atleast_1d(release['start_depth'])[i], num)

        if model.vert_migration == True:
            depth = np.repeat(np.atleast_1d(model.vert_array[date_range[0].tolist().hour]),
                                            num)

        particle = Particle(num, lat, lon, depth)
        transformed = grid.tgt_crs.transform_points(grid.src_crs,
                                                    lon, lat, depth).T ###
        particle.x, particle.y, particle.depth = transformed[:3].astype(np.float32)###
        particle.filepath = get_filepath(date_range[0], model.model,
                                         model.submodel, model.data_dir)
        particle.timestamp = date_range[0]
        particle = get_physical(particle, grid, model)
        for j in range(len(date_range)):
            #particle.timestamp = date_range[j]
            particle.filepath = get_filepath(particle.timestamp, model.model,
                                             model.submodel, model.data_dir)
            # if j.tolist().hour == 0:
            #    print('getting physical data for ' + str(j))
            particle_arr = np.array([particle.x, particle.y, particle.depth,
                                     particle.u, particle.v, particle.w,
                                     particle.temp, particle.sal]).T ###
            trajectory[:, j, :] = particle_arr
            particle = force_particle(particle, grid, model)

        transformed = grid.src_crs.transform_points(grid.tgt_crs,
                                                    trajectory[..., 0],
                                                    trajectory[..., 1],
                                                    trajectory[..., 2])
        lons = transformed[..., 0]
        lats = transformed[..., 1]
        depths = transformed[..., 2]

        # write the data to netcdf
        nc_data = xr.Dataset({'lat': (['release', 'time'], lats),
                              'lon': (['release', 'time'], lons),
                              'depth': (['release', 'time'], depths),
                              'u': (['release', 'time'], trajectory[..., 3]),
                              'v': (['release', 'time'], trajectory[..., 4]),
                              'w': (['release', 'time'], trajectory[..., 5]),
                              'temp': (['release', 'time'], trajectory[..., 6]),
                              'sal': (['release', 'time'], trajectory[..., 7])},
                             coords={'release': np.arange(0, num),
                                     'time': date_range})

        nc_data.to_netcdf('output/'
                          + np.atleast_1d(release['particle_id'])[i].astype(str)
                          + '_' + model.output_file)
        nc_data.close()