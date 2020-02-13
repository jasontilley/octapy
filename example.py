import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import octapy

#create the data and output directories if they don't already exist
#os.mkdir('data')
#os.mkdir('output')

# initialize the model
release_file = 'release.csv'
model = octapy.tracking.Model(release_file, 'HYCOM', 'GOMl0.04/expt_31.0',
                              interp='idw', leafsize=3)

# the data has one file for each hour (use minutes as unit)
data_start = np.datetime64('2010-06-09')
data_stop = np.datetime64('2010-07-07')
model.data_freq = np.timedelta64(1440,'m')

model.data_date_range = np.arange(data_start, data_stop, step=model.data_freq)

# enter the timestep as a np.timedelta64
# currently, it is best that timestep <= data_date_range
model.timestep = np.timedelta64(360,'m')

model.depth = 15
model.output_file = 'output.csv'

#download the data
octapy.tracking.download_data(model)

# initialize the grid
grid = octapy.Grid(model)

# run the model
octapy.tracking.run_model(model, grid)

# plot the output
output_files = glob.glob('*output.csv')
extent = octapy.tools.get_extent(grid)
octapy.tools.plot_csv_output(output_files, extent=extent, step=4)
