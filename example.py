from os.path import expanduser
import glob
import numpy as np
import octapy

# create the data and output directories if they don't already exist
# os.mkdir('data')
# os.mkdir('output')

# initialize the model
data_dir = expanduser('~') + '/Desktop/data'
release_file = 'release.csv'
model = octapy.tracking.Model(release_file, 'HYCOM', 'GOMl0.04/expt_31.0',
                              data_dir=data_dir, interp='idw', leafsize=3)

# the data has one file for each hour (use minutes as unit)
# BUG - must run separately for each year date range
data_start = np.datetime64('2010-06-09')
# make sure to add an additional day of data for proper time interpolation
data_stop = np.datetime64('2010-07-08')
model.data_freq = np.timedelta64(60, 'm')
model.data_timestep = np.timedelta64(1440, 'm')

# enter the timestep as a np.timedelta64
# currently, it is best that timestep <= data_date_range
# be careful to set your data_freq to match server data time step and to set
# your model.timestep to match you desired timestep. Set the timestep before
# you set the data_date_range! Check the dates in your files!
# model timestep must be factor of 60 min
model.timestep = np.timedelta64(60, 'm')
model.data_date_range = np.arange(data_start, data_stop,
                                  step=model.data_timestep)

model.depth = 15
model.output_file = 'output.nc'

# download the data
octapy.tracking.download_hycom_data(model)

# initialize the grid
grid = octapy.Grid(model)

# run the model
octapy.tracking.run_2d_model(model, grid)

# plot the output
# output_files = glob.glob('*output.nc')
output_files = ['output/88589_output.nc']
extent = octapy.tools.get_extent(grid)
octapy.tools.plot_netcdf_output(output_files, extent=extent, step=1,
                                plot_type='lines',
                                drifter='output/drifter_88589_output.csv')


# example for initializing a length-1 particle
# from octapy.tracking import Particle, transform
# from octapy.tools import get_filepath
# from octapy.get_data_at_index import get_data_at_index
# particle = Particle(28., -88., 15., np.datetime64('2010-06-08', 's'))
# particle.x, particle.y = transform(grid.src_crs, grid.tgt_crs,
#                                    np.array([particle.lon]),
#                                    np.array([particle.lat]))
# particle.x = np.array([particle.x])
# particle.y = np.array([particle.y])
# particle.filepath = get_filepath(particle.timestamp, model.model,
#                                      model.submodel, model.data_dir)

# skill example for drifter 88589
data_dir = expanduser('~') + '/Desktop/data'
release_file = 'release.csv'
model = octapy.tracking.Model(release_file, 'HYCOM', 'GOMl0.04/expt_31.0',
                              data_dir=data_dir, interp='idw', leafsize=3)
data_start = np.datetime64('2010-06-09')
data_stop = np.datetime64('2010-07-08')
model.data_freq = np.timedelta64(60, 'm')
model.data_timestep = np.timedelta64(1440, 'm')
model.timestep = np.timedelta64(60, 'm')
model.data_date_range = np.arange(data_start, data_stop,
                                  step=model.data_timestep)
model.depth = 15
model.output_file = 'output.nc'
grid = octapy.Grid(model)

drifter_file = 'gom_drifters.csv'
drifter_id = 88589
octapy.tools.build_skill_release(drifter_file, model)
model.release_file = 'release_drifter_' + str(drifter_id) + '.csv'
octapy.tracking.run_2d_model(model, grid)

# plot the skill output tracks
output_files = glob.glob('output/' + str(drifter_id) + '_*.nc')
# remove the full track from the plot
output_files.remove('output/' + str(drifter_id) + '_output.nc')
extent = octapy.tools.get_extent(grid)
octapy.tools.plot_netcdf_output(output_files, extent=extent, step=1,
                                plot_type='lines',
                                drifter='output/drifter_' + str(drifter_id)
                                        + '_output.csv')

# run the skill analysis
# next try 10 day
from octapy.tools import *
skill_files = sorted(output_files)
date_range = model.data_date_range

# return particle id, trajectory_length, separation distance, and c
skill = run_skill_analysis(drifter_file, drifter_id, skill_files, date_range,
                           grid, period=pd.Timedelta('3 Days'),
                           data_freq=pd.Timedelta('60 minutes'))


# run backward
data_dir = expanduser('~') + '/Desktop/data'
release_file = 'release_back.csv'
model = octapy.tracking.Model(release_file, 'HYCOM', 'GOMl0.04/expt_31.0',
                              direction=-1, data_dir=data_dir,
                              interp='idw', leafsize=3)
# make sure to add an additional day of data for proper time interpolation
data_start = np.datetime64('2010-07-06')
data_stop = np.datetime64('2010-06-08')
model.data_timestep = np.timedelta64(1440, 'm')
model.timestep = np.timedelta64(60, 'm')
# must have date range with negative timestep
model.data_date_range = np.arange(data_start, data_stop,
                                  step=-model.data_timestep)

model.depth = 15
model.output_file = 'back_output.nc'

# download the data
octapy.tracking.download_data(model)

# initialize the grid
grid = octapy.Grid(model)

# run the model
octapy.tracking.run_2d_model(model, grid)

output_files = ['output/88589_back_output.nc']
drifter_id = 88589
extent = octapy.tools.get_extent(grid)
octapy.tools.plot_netcdf_output(output_files, extent=extent, step=1,
                                plot_type='lines', drifter='output/drifter_'
                                + str(drifter_id) + '_back_output.csv')