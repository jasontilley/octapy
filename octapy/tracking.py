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
from scipy.interpolate import interpn
from .tools import *


class Model():
    ''' A particle tracking Model object
    
    Keyword arguments:
    release_file -- a file containing the particle coordinates and release times
    model -- name string of the ocean model used for input data (e.g, 'HYCOM')
    submodel -- name string of the submodel and/or experiment used for input
                data (e.g, 'GOMl0.04/expt_31.0')
    grid -- location of the model Grid class object
    data_dir -- data directory path
    dir -- forcing direction through time. Must be 'forward' or 'backward'
    dims -- dimensionality of the model, must be 2 or 3
    depth -- forcing depth if the model is 2-dimensional
    extent -- a list of extent coordinates as [minlat, maxlat, minlon, maxlon]
    data_date_range -- a pandas.date_range object containing the starting time,
                       ending time, and the frequency of the input data
    timestep -- a pandas.tseries.offsets object representing the timestep of the
                tracking model (e.g., pandas.tseries.offsets.Hour(1))
    vert_migration -- if True, the particle will undergo daily vertical
                      migrations which will override w velocities
    vert_array -- an array of length 24 representing the depth of a particle
                  over a 24-hour period when vert_mirgration is set to True
    output_file = base output file name
    output_freq = a pandas.tseries.offsets object representing how often the
                  particle data will be output to the output file
                  (e.g., pandas.tseries.offsets.Hour(1))
                  
    Returns:
    A Model object
    
    '''
    
    def __init__(self, release_file=None, model=None, submodel=None,
                 grid=None, data_dir='data', dir='forward', dims=2, depth=None,
                 extent=None, data_date_range=None,
                 timestep = pd.tseries.offsets.Hour(1), vert_array=None,
                 output_file=None, output_freq=pd.tseries.offsets.Hour(1)):
                 
        self.release_file = release_file
        self.model = model
        self.submodel = submodel
        self.experiment = experiment
        self.grid = None
        self.data_dir = data_dir
        self.dir = dir
        self.dims = dims
        self.depth = depth
        self.extent = extent
        self.data_date_range = data_date_range
        self.timestep = timestep
        self.vert_migration = False
        self.vert_array = vert_array
        self.output_file = output_file
        self.output_freq = output_freq
        
    def download_data(self):
    
        if self.submodel == 'GOMl0.04/expt_31.0':
            year = self.data_date_range.year[0]
            dates = pd.date_range('1/1/' + str(year), '1/1/' + str(year + 1),
                                  freq=self.data_date_range.freq)
            depths = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0,
                               50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0,
                               150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0,
                               700.0, 800.0, 900.0,1000.0, 1100.0, 1200.0,
                               1300.0, 1400.0, 1500.0, 1750.0, 2000.0, 2500.0,
                               3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0])
            self.model_dir = ('http://tds.hycom.org/thredds/dodsC/' +
                              self.submodel + '/' + str(year) + '/hrly')
                              
        for date in (self.data_date_range + 1):
            dt = pd.to_datetime(date)
            time_idx = np.where(dates == dt)[0][0]
            
            if self.depth != None:
                depth_idx = np.where(depths == self.depth)[0][0]
            
            ncfile = get_filepath(date, self)
                      
            if path.isfile(ncfile):
                print('File already exists!')
                continue
                
            if self.dims == 3:
                dim_str = '[' + str(time_idx) + '][0:1:39][0:1:384][0:1:540]'
                query_str = ('?Depth[0:1:39],Latitude[0:1:384],'
                             + 'Longitude[0:1:540],MT[' + str(time_idx) + '],'
                             + 'u' + dim_str + ',' + 'v' + dim_str + ','
                             + 'w_velocity' + dim_str + ','
                             + 'temperature' + dim_str + ',' + 'salinity'
                             + dim_str)
                             
            if self.dims == 2:
                dim_str = ('[' + str(time_idx) + '][' + str(depth_idx) +
                           '][0:1:384][0:1:540]')
                query_str = ('?Depth[' + str(depth_idx)
                             + '],Latitude[0:1:384],' + 'Longitude[0:1:540],MT['
                             + str(time_idx) + '],' + 'u' + dim_str + ',' + 'v'
                             + dim_str + ',' + 'w_velocity' + dim_str + ','
                             + 'temperature' + dim_str + ',' + 'salinity'
                             + dim_str)
            
            print('Writing: ' + ncfile)
            data = xr.open_dataset(self.model_dir + query_str)
            data.to_netcdf(ncfile)
                        
                        
class Particle():
    ''' A Particle object
    
    Keyword arguments:
    model -- Model instance which the Particle instance belongs
    lat -- Latitude of the particle
    lon -- Longitude of the particle
    depth -- Depth of the particle in meters
    timestamp -- a pandas Timestamp instance
    
    Returns:
    A Particle object
    
    Attributes:
    x -- the x coordinate in meters
    y -- the y coordinate in meters
    u -- the u-velocity (eastward velocity) at the particle's location
    v -- the v-velocity (northward velocity) at the particle's location
    w -- the w-velocity (upward velocity) at the particle's location
    temp -- the temperature at the particle's location
    sal -- the salinity at the particle's location
    filepath -- the expected file path given the particles timestamp
    
    '''
    
    def __init__(self, model, lat, lon, depth, timestamp):
    
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.timestamp = timestamp
        self.x, self.y = model.grid.tgt_crs.transform_point(self.lon, self.lat,
                                                            model.grid.src_crs)
        self.u = None
        self.v = None
        self.w = None
        self.temp = None
        self.sal = None
        self.filepath = get_filepath(self.timestamp, model)
        

class Grid():
    ''' A Grid object. You must have already downloaded the data into the data
    directory.
    
    Keyword arguments:
    model -- Model instance to which the Grid instance will belong
    
    Returns:
    A Grid object
    
    Attributes:
    src_crs -- a cartopy.crs projection object representing the data's source
               projection
    src_crs -- a cartopy.crs projection object representing the model's target
               projection
    file -- Name of the file from which the grid was produced.
    lats -- Latitudes of the grid
    lons -- Longitudes of the grid
    depths -- Depths of the grid
    x -- the x coordinates of the grid in meters
    y -- the y coordinates of the grid in meters
    
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

def interp_physical(particle, model):

    particle.u = interp3d(particle, model, 'u')
    particle.v = interp3d(particle, model, 'v')
    particle.temp = interp3d(particle, model, 'temperature')
    particle.sal = interp3d(particle, model, 'salinity')
    particle.w = None
    if model.dims == 3:
        particle.w = interp3d(particle, model, 'w_velocity')
    return(particle)


def get_physical(particle, model):

    if path.isfile(particle.filepath):
        particle = interp_physical(particle, model)
        
    # else find the two surrounding files
    else:
    
        for i in range(0,24):
            new_timestamp = (particle.timestamp.round('h')
                             - pd.Timedelta(str(i) + ' hours'))
            file1 = get_filepath(new_timestamp, model)
            if path.isfile(file1):
                particle1 = deepcopy(particle)
                particle1.timestamp = new_timestamp
                particle1.filepath = file1
                break
                
        for i in range(1,24):
            new_timestamp = (particle.timestamp.round('h')
                             + pd.Timedelta(str(i) + ' hours'))
            file2 = get_filepath(new_timestamp, model)
            if path.isfile(file2):
                particle2 = deepcopy(particle)
                particle2.timestamp = new_timestamp
                particle2.filepath = file2
                break
                
        particle1 = interp_physical(particle1, model)
        particle2 = interp_physical(particle2, model)
        particle = interp_for_time(model, particle, particle1, particle2)
    return(particle)
        
        
def interp3d(particle, model, variable_str, method='linear'):

    if model.dims == 2:
        points = np.array([model.grid.x, model.grid.y])
        data = xr.open_dataset(particle.filepath)[variable_str].squeeze('MT')
        data = data.data.T
        value = interpn(points, data, (particle.x, particle.y),
                        method=method)
                        
    if model.dims == 3:
        points = np.array([model.grid.x, model.grid.y, model.grid.depths])
        data = xr.open_dataset(particle.filepath)[variable_str].squeeze('MT')
        data = data.data.T
        value = interpn(points, data, (particle.x, particle.y, particle.depth),
                        method=method)
                        
    return(value.item())


def interp_for_time(model, particle, particle1, particle2, method='linear'):

    points = np.array([particle1.timestamp.to_julian_date(),
                          particle2.timestamp.to_julian_date()])
    xi = particle.timestamp.to_julian_date()
    
    for i in ['u', 'v', 'temp', 'sal']:
        value = np.interp(xi, points, np.array([particle1.__getattribute__(i),
                                                particle2.__getattribute__(i)]))
        setattr(particle, i, value)
        
    if model.dims == 3:
        value = np.interp(xi, points, np.array([particle1.w, particle2.w]))
        setattr(particle, 'w', value)
        
    return(particle)
    
    
def force_particle(particle, model):

    if model.dir == 'forward':
        i = 1
        
    if model.dir == 'backward':
        i = -1
        
    particle.x = particle.x + model.timestep.delta.seconds * particle.u * i
    particle.y = particle.y + model.timestep.delta.seconds * particle.v * i
    
    transformed = model.grid.src_crs.transform_point(particle.x, particle.y,
                                                     model.grid.tgt_crs)
    particle.lat = transformed[1]
    particle.lon = transformed[0]
    
    if model.dims == 3:
        particle.depth = (particle.depth + model.timestep.delta.seconds
                          * particle.w * i)
                          
    particle.timestamp = particle.timestamp + model.timestep * i
    particle.filepath = get_filepath(particle.timestamp, model)
    particle = get_physical(particle, model)
    return(particle)


def add_row_to_df(df, particle):

    row = pd.Series(index=df.columns)
    
    for i in row.index:
        row.loc[i] = getattr(particle, i)
        
    df = df.append(row, ignore_index=True)
    return(df)


def run_model(model):

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
                               freq=model.timestep):
            particle = Particle(model, lat, lon, depth, j)
            if j.minute == 0:
                print('getting physical data for ' + j.ctime())
            particle = get_physical(particle, model)
            trajectory = add_row_to_df(trajectory, particle)
            particle = force_particle(particle, model)
            lat = particle.lat
            lon = particle.lon
            depth = particle.depth
            
        trajectory.to_csv(release.iloc[i]['particle_id'] + '_'
                          + model.output_file, index=False)
            
def interp_inverse_distance(dx, dy, re, kt, index)
        
#dr=sqrt(dx^2+dy^2)
#wt=1./dr              ;.....inverse distance
#;...wt=exp(-dr^2/re^2)     ;......gaussian
#u1=total(wt*u(kt,index))/total(wt)
#v1=total(wt*v(kt,index))/total(wt)
#;
#u2=total(wt*u(kt+1,index))/total(wt)
#v2=total(wt*v(kt+1,index))/total(wt)
#un=timefact*(u2-u1)+u1
#vn=timefact*(v2-v1)+v1



        

