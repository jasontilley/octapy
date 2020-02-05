import glob
import pandas as pd
import octapy

# initialize the model
release_file = 'release.csv'
model = octapy.tracking.Model(release_file, 'HYCOM', 'GOMl0.04/expt_31.0')
model.data_date_range = pd.date_range('6/9/10', '7/7/10', freq='1H')
model.timestep = pd.tseries.offsets.Minute(6)
model.depth = 15
model.output_file = 'output.csv'

#download the data
model.download_data()

# initialize the grid
model.grid = octapy.Grid(model)

# run the model
octapy.tracking.run_model(model)

# plot the output
output_files = glob.glob('output/*output.csv')
extent = octapy.tools.get_extent(model)
octapy.tools.plot_csv_output(output_files, extent=extent, step=240)

