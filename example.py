import glob
import numpy as np
import octapy

# create the data and output directories if they don't already exist
# os.mkdir('data')
# os.mkdir('output')

# initialize the model
release_file = 'release.csv'
model = octapy.tracking.Model(release_file, 'HYCOM', 'GOMl0.04/expt_31.0',
                              interp='idw', leafsize=3)

# the data has one file for each hour (use minutes as unit)
data_start = np.datetime64('2009-05-01')
data_stop = np.datetime64('2009-05-31')
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
octapy.tracking.download_data(model)

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

# skill example for drifter 75196
release_file = 'release.csv'
model = octapy.tracking.Model(release_file, 'HYCOM', 'GOMl0.04/expt_31.0',
                              interp='idw', leafsize=3)
data_start = np.datetime64('2009-05-01')
data_stop = np.datetime64('2009-05-31')
model.data_freq = np.timedelta64(60, 'm')
model.data_timestep = np.timedelta64(1440, 'm')
model.timestep = np.timedelta64(60, 'm')
model.data_date_range = np.arange(data_start, data_stop,
                                  step=model.data_timestep)
model.depth = 15
model.output_file = 'output.nc'
grid = octapy.Grid(model)

drifter_file = 'gom_drifters.csv'
drifter_id = 75196
octapy.tools.build_skill_release(drifter_file, model)
model.release_file = 'release_drifter_75196.csv'
octapy.tracking.run_2d_model(model, grid)

# plot the skill output tracks
output_files = glob.glob('output/75196_*.nc')
# remove the full track from the plot
output_files.remove('output/75196_output.nc')
extent = octapy.tools.get_extent(grid)
octapy.tools.plot_netcdf_output(output_files, extent=extent, step=1,
                                plot_type='lines',
                                drifter='output/drifter_75196_output.csv')



# pretty plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from owslib.wmts import WebMapTileService

URL = 'http://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi'
wmts = WebMapTileService(URL)
layer = 'BlueMarble_NextGeneration'
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(extent)
ax.add_wmts(wmts, layer, wmts_kwargs={'time': '2016-02-05'})
ax.coastlines(resolution='10m')