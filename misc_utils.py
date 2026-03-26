import numpy as np

def auto_split_range(data,cmap='bwr',force_range=None):
    output = {
        'vmin':None,
        'vmax':None,
        'cmap':cmap
    }

    data_max = np.max(data)
    data_min = np.min(data)

    abs_range = np.max([data_max,np.abs(data_min)])

    if force_range is not None:
        abs_range = force_range
        
    output['vmin'] = -1 * abs_range
    output['vmax'] = abs_range

    return output
