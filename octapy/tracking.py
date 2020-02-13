#octapy – Ocean Connectivity and Tracking Algorithms
#Copyright (C) 2020  Jason Tilley

import enum
import glob
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from os import path
from copy import deepcopy
from numba import jit, jitclass, types, typed
from scipy.interpolate import interpn
from scipy.spatial import cKDTree
from .tools import *
from .data import Data
from . import interp_idw


@jitclass([('release_file', types.unicode_type),
           ('model', types.unicode_type),
           ('submodel', types.unicode_type),
           ('data_dir', types.unicode_type),
           ('dir', types.unicode_type),
           ('dims', types.uint8),
           ('depth', types.optional(types.float32)),
           ('extent', types.optional(types.ListType(types.float32))),
           ('data_date_range', types.optional(types.NPDatetime('m')[:])),
           ('data_freq', types.NPTimedelta('m')),
           ('timestep', types.optional(types.NPTimedelta('m'))),
           ('interp', types.unicode_type),
           ('leafsize', types.int32),
           ('vert_migration', types.boolean),
           ('vert_array', types.optional(types.float32[24])),
           ('output_file', types.optional(types.unicode_type)),
           ('output_freq', types.NPTimedelta('m'))])
class Model():
    ''' A particle tracking Model object
    
    Keyword arguments:
    release_file -- a file containing the particle coordinates and release times
    model -- name string of the ocean model used for input data (e.g, 'HYCOM')
    submodel -- name string of the submodel and/or experiment used for input
                data (e.g, 'GOMl0.04/expt_31.0')
    data_dir -- data directory path
    dir -- forcing direction through time. Must be 'forward' or 'backward'
    dims -- dimensionality of the model, must be 2 or 3
    data_vars -- an numpy.ndarray of the variable names in the data file
    depth -- forcing depth if the model is 2-dimensional
    extent -- a list of extent coordinates as [minlat, maxlat, minlon, maxlon]
    data_date_range -- a numpy.ndarray object of numpy.datetime64 objects
                       containing the dates of the input data
    timestep -- a numpy.timedelta64 object representing the timestep of the
                tracking model in minutes (e.g., np.timedelta64(60,'m'))
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
    
    '''
    
    def __init__(self, release_file=None, model=None, submodel=None,
                 data_dir='data', dir='forward', dims=2,
                 #data_vars = None,
                 depth=None, extent=None,
                 data_date_range=None, data_freq = np.timedelta64(60,'m'),
                 timestep=np.timedelta64(60,'m'), interp='linear', leafsize=9,
                 vert_migration=False, vert_array=None, output_file=None,
                 output_freq=np.timedelta64(60,'m')):
                 
        self.release_file = release_file
        self.model = model
        self.submodel = submodel
        self.data_dir = data_dir
        self.dir = dir
        self.dims = dims
        self.depth = depth
        self.extent = extent
        self.data_date_range = data_date_range
        self.timestep = timestep
        self.interp = interp
        self.leafsize = leafsize
        self.vert_migration = vert_migration
        self.vert_array = vert_array
        self.output_file = output_file
        self.output_freq = output_freq
         

class Grid():
    ''' A Grid object. You must have already downloaded the data into the data
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
    
    '''
    
    def __init__(self, model):
    
        self.src_crs = ccrs.LambertCylindrical()
        self.tgt_crs = ccrs.Mercator(central_longitude=-87.200012,
                                     min_latitude=18.091648,
                                     max_latitude=31.960648)
        self.file = glob.glob(model.data_dir + '/*')[0]
        data = xr.open_dataset(self.file)
        self.lats = data['Latitude'].data
        self.lons = data['Longitude'].data
        self.depths = data['Depth'].data
        self.lons, self.lats = np.meshgrid(self.lons, self.lats)
        transformed = self.tgt_crs.transform_points(self.src_crs, self.lons,
                                                    self.lats)
        self.lons = self.lons[0]
        self.lats = self.lats.T[0]
        self.x = transformed[0,:,0]
        self.y = transformed[:,0,1]
        x, y = np.meshgrid(self.x, self.y)
        
        if model.dims == 2:
            self.points = np.array([self.x, self.y])
            
        if model.dims == 3:
            self.points = np.array([self.x, self.y, self.depths])
        
        xy_coords = np.array([x.ravel(), y.ravel()]).T
        self.tree = cKDTree(xy_coords, leafsize=model.leafsize)


@jitclass([('lat', types.float32),
           ('lon', types.float32),
           ('depth', types.float32),
           ('timestamp', types.NPDatetime('s')),
           ('x', types.optional(types.float32)),
           ('y', types.optional(types.float32)),
           ('u', types.optional(types.float32)),
           ('v', types.optional(types.float32)),
           ('w', types.optional(types.float32)),
           ('temp', types.optional(types.float32)),
           ('sal', types.optional(types.float32)),
           ('filepath', types.optional(types.unicode_type))])
class Particle():
    ''' A Particle object
    
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
    
    '''
    
    def __init__(self, lat, lon, depth, timestamp):
    
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.timestamp = timestamp
        self.x = None
        self.y = None
        self.u = None
        self.v = None
        self.w = None
        self.temp = None
        self.sal = None
        self.filepath = None


#@jitclass([('datetime64', types.optional(types.NPDatetime('m'))),
#           ('u', types.optional(types.float32[:,:,:,:])),
#           ('v', types.optional(types.float32[:,:,:,:])),
#           ('w', types.optional(types.float32[:,:,:,:])),
#           ('temp', types.optional(types.float32[:,:,:,:])),
#           ('sal', types.optional(types.float32[:,:,:,:]))])
#class Data():
#
#    def __init__(self, particle):
#
#        self.datetime64 = None
#        self.u = None
#        self.v = None
#        self.w = None
#        self.temp = None
#        self.sal = None
    

def transform(src_crs, tgt_crs, lon, lat):
    x, y = tgt_crs.transform_point(lon, lat, src_crs)
    return(x,y)


def download_data(model):

    if model.submodel == 'GOMl0.04/expt_31.0':
        year = model.data_date_range[0].item().year
        dates = pd.date_range('1/1/' + str(year), '1/1/' + str(year + 1),
                              freq=pd.Timedelta(model.data_freq))
        depths = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0,
                           50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0,
                           150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0,
                           700.0, 800.0, 900.0,1000.0, 1100.0, 1200.0,
                           1300.0, 1400.0, 1500.0, 1750.0, 2000.0, 2500.0,
                           3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0])
        model_dir = ('http://tds.hycom.org/thredds/dodsC/' +
                     model.submodel + '/' + str(year) + '/hrly')
    
    model.data_date_range = np.append(model.data_date_range,
                                      model.data_date_range[-1]
                                      + model.data_freq)
    for date in model.data_date_range:
        #dt = pd.to_datetime(date, 'm')
        
        try:
            time_idx = np.where(dates == date)[0][0]
            
        except IndexError:
            continue
        
        if model.depth != None:
            depth_idx = np.where(depths == model.depth)[0][0]
        
        ncfile = get_filepath(date.astype('datetime64[s]'), model)
                  
        if path.isfile(ncfile):
            print('File already exists!')
            continue
            
        if model.dims == 3:
            dim_str = '[' + str(time_idx) + '][0:1:39][0:1:384][0:1:540]'
            query_str = ('?Depth[0:1:39],Latitude[0:1:384],'
                         + 'Longitude[0:1:540],MT[' + str(time_idx) + '],'
                         + 'u' + dim_str + ',' + 'v' + dim_str + ','
                         + 'w_velocity' + dim_str + ','
                         + 'temperature' + dim_str + ',' + 'salinity'
                         + dim_str)
                         
        if model.dims == 2:
            dim_str = ('[' + str(time_idx) + '][' + str(depth_idx) +
                       '][0:1:384][0:1:540]')
            query_str = ('?Depth[' + str(depth_idx)
                         + '],Latitude[0:1:384],' + 'Longitude[0:1:540],MT['
                         + str(time_idx) + '],' + 'u' + dim_str + ',' + 'v'
                         + dim_str + ',' + 'w_velocity' + dim_str + ','
                         + 'temperature' + dim_str + ',' + 'salinity'
                         + dim_str)
        
        print('Writing: ' + ncfile)
        data = xr.open_dataset(model_dir + query_str)
        data.to_netcdf(ncfile)


def get_physical(particle, grid, model):

    if path.isfile(particle.filepath):
        particle = interp3d(particle, grid, model)
        
    # else find the two surrounding files
    else:
    
        for i in range(0,24):
            new_timestamp = (pd.Timestamp(particle.timestamp).round('h')
                             - pd.Timedelta(str(i) + ' hours'))
            file1 = get_filepath(np.datetime64(new_timestamp, 's'), model)
            if path.isfile(file1):
                particle1 = deepcopy(particle)
                particle1.timestamp = new_timestamp
                particle1.filepath = file1
                break
                
        for i in range(1,24):
            new_timestamp = (pd.Timestamp(particle.timestamp).round('h')
                             + pd.Timedelta(str(i) + ' hours'))
            file2 = get_filepath(np.datetime64(new_timestamp, 's'), model)
            if path.isfile(file2):
                particle2 = deepcopy(particle)
                particle2.timestamp = new_timestamp
                particle2.filepath = file2
                break
                
        particle1 = interp3d(particle1, grid, model)
        particle2 = interp3d(particle2, grid, model)
        particle = interp_for_time(model, particle, particle1, particle2)
    return(particle)
        
        
def interp3d(particle, grid, model):

    if model.interp == 'idw':
    
        particle = interp_idw(particle, grid, model, power=1.0)
    
    else:
    
        rootgrp = xr.open_dataset(particle.filepath)
        data = Data(particle)
    
        if model.dims == 2:
        
            nc_vars = ['u', 'v', 'temperature', 'salinity']
            part_vars = ['u', 'v', 'temp', 'sal']
            
            for i, j in zip(nc_vars, part_vars):
            
                setattr(data, j, rootgrp[i].values)
                value = interpn(grid.points, data.__getattribute__(j)[0].T,
                                (particle.x, particle.y),
                                method=model.interp)
                setattr(particle, j, value.item())
            
        if model.dims == 3:
            nc_vars = ['u', 'v', 'w_velocity', 'temperature', 'salinity']
            part_vars = ['u', 'v', 'w', 'temp', 'sal']
                        
    return(particle)


def interp_for_time(model, particle, particle1, particle2, method='linear'):

    points = np.array([particle1.timestamp.to_julian_date(),
                          particle2.timestamp.to_julian_date()])
    xi = particle.timestamp.to_julian_date()
    
    for i in model.part_vars:
        value = np.interp(xi, points, np.array([particle1.__getattribute__(i),
                                                particle2.__getattribute__(i)]))
        setattr(particle, i, value)
        
    return(particle)
    
    
def force_particle(particle, grid, model):

    if model.dir == 'forward':
        i = 1
        
    if model.dir == 'backward':
        i = -1
    
    particle.x = particle.x + model.timestep.item().seconds * particle.u * i
    particle.y = particle.y + model.timestep.item().seconds * particle.v * i
    
    transformed = grid.src_crs.transform_point(particle.x, particle.y,
                                               grid.tgt_crs)
    particle.lat = transformed[1]
    particle.lon = transformed[0]
    
    if model.dims == 3:
        particle.depth = (particle.depth + model.timestep.item().seconds
                          * particle.w * i)
                          
    particle.timestamp = particle.timestamp + model.timestep * i
    particle.filepath = get_filepath(particle.timestamp, model)
    particle = get_physical(particle, grid, model)
    return(particle)


def add_row_to_df(df, particle):

    row = pd.Series(index=df.columns)
    
    for i in row.index:
        row.loc[i] = getattr(particle, i)
        
    df = df.append(row, ignore_index=True)
    return(df)


def run_model(model, grid):

    release = pd.read_csv(model.release_file)
    release['start_time'] = pd.to_datetime(release['start_time'])
    release['particle_id'] = release['particle_id'].astype(str)
    
    for i in release.index:
        print('getting particle info for particle '
              + release.iloc[i]['particle_id'])
        trajectory = pd.DataFrame(columns=['timestamp', 'lat', 'lon', 'depth',
                                           'u', 'v', 'w', 'temp', 'sal'])
        lat = release.iloc[i]['start_lat']
        lon = release.iloc[i]['start_lon']
        depth = release.iloc[i]['start_depth']
        
        for j in pd.date_range(release.iloc[i]['start_time'],
                               release.iloc[i]['start_time']
                               + pd.Timedelta(str(release.iloc[i]['days'])
                               + ' days'), # - model.timestep,
                               freq=pd.Timedelta(model.timestep)):
            particle = Particle(lat, lon, depth, np.datetime64(j, 's'))
            particle.x, particle.y = transform(grid.src_crs, grid.tgt_crs,
                                               particle.lon, particle.lat)
            particle.filepath = get_filepath(particle.timestamp, model)
            if j.minute == 0:
                print('getting physical data for ' + j.ctime())
            particle = get_physical(particle, grid, model)
            trajectory = add_row_to_df(trajectory, particle)
            particle = force_particle(particle, grid, model)
            lat = particle.lat
            lon = particle.lon
            depth = particle.depth
            
        trajectory.to_csv(release.iloc[i]['particle_id'] + '_'
                          + model.output_file, index=False)
            
            
#def interp_idw(particle, grid, model, power=1.0):
#
#    distances, indices = grid.tree.query([(particle.x,  particle.y)],
#                                         k=model.leafsize)
#    weights = 1. / distances ** power
#
#    if model.dims == 2:
#        nc_vars = ['u', 'v', 'temperature', 'salinity']
#        part_vars = ['u', 'v', 'temp', 'sal']
#
#    if model.dims == 3:
#        nc_vars = ['u', 'v', 'w_velocity', 'temperature', 'salinity']
#        part_vars = ['u', 'v', 'w', 'temp', 'sal']
#
#    rootgrp = xr.open_dataset(particle.filepath)
#    data = Data(particle)
#
#    for i, j in zip(nc_vars, part_vars):
#        setattr(data, j, rootgrp[i].values)
#        values = data.__getattribute__(j).ravel()[indices][0]
#        value = (weights * values).sum() / weights.sum()
#        setattr(particle, j, value)
#
#    return(particle)

