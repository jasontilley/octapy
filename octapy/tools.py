import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from os.path import splitext

def get_filepath(timestamp, model):
    ''' Get the filename for a given timestep
    
    Keyword arguments:
    time -- A pandas Timestamp instance
    model -- the oceanographic model being used
    submodel -- the submodel being used
    '''
    date_str = timestamp.strftime('%Y%m%d%H%M%S')
    filepath = (model.data_dir + '/' + (date_str + '.' + model.model + '.' +
                                        model.submodel.replace('/', '.') +
                                        '.nc'))
    return(filepath)
    
    
def get_extent(model):
    extent = [model.grid.lons.min(), model.grid.lons.max(),
              model.grid.lats.min(), model.grid.lats.max()]
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
