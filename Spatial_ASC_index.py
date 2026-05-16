#### Spatial Air-Sea coupling strength index ####
# ASC index #

# Module
import numpy as np
import xarray as xr

from numba import jit
from joblib import Parallel, delayed


# Prepare 'daily anomaly' SST and T2m data for the summer season.
anos2=mera2['sst'] 
anot2=mera2['t2m'] # Only ocean area


@jit(nopython=True)
def calculate_overlap_fast(g_t, g_s, quantile_t, quantile_s):
    # Extreme-event overlap calculation accelerated with Numba
    mask_t = g_t > quantile_t
    mask_s = g_s > quantile_s
    BZ = np.sum(mask_t & mask_s)
    BM = np.sum(mask_t)
    return BZ / BM if BM > 0 else np.nan

# Optimized calculation using vectorized distance computation
def calculate_g_series_optimized(data_array, ref_time_idx):
    # Reference map selection
    map_ref = data_array.isel(time=ref_time_idx)
    target_time = data_array.time.isel(time=ref_time_idx).values
    
    # Exclude the reference time step
    mask = data_array.time.values != target_time
    data_array2 = data_array.isel(time=mask)
    
    # Euclidean distance calculation
    squared_diff = (data_array2 - map_ref)**2
    sum_of_squares = squared_diff.sum(dim=['latitude', 'longitude'], skipna=True)
    dist = np.sqrt(sum_of_squares)
    g_series = np.where(dist > 0, -np.log(dist), np.inf)
    
    return g_series


# Parallel processing 
from joblib import Parallel, delayed

def process_single_time(i, anos_summer, anot_summer):
    # Compute g_series for each variable
    try:
        g_r_s = calculate_g_series_optimized(anos_summer, i)
        g_r_t = calculate_g_series_optimized(anot_summer, i)
        
        # Convert to NumPy arrays
        g_t_arr = np.asarray(g_r_t)
        g_s_arr = np.asarray(g_r_s)

        # 98th percentile threshold calculation
        q_t = np.quantile(g_t_arr, 0.98)
        q_s = np.quantile(g_s_arr, 0.98)
        
        alp = calculate_overlap_fast(g_t_arr, g_s_arr, q_t, q_s)
    except:
        alp = np.nan
        
    return alp

# Parallelized overlap calculation without lag
def process_no_lag_parallel(anos, anot, n_jobs=12):

    # Summer season selection (JJA)
    summer_mask_anos = anos.time.dt.month.isin([6, 7, 8])
    summer_mask_anot = anot.time.dt.month.isin([6, 7, 8])

    anos_summer = anos.isel(time=summer_mask_anos)
    anot_summer = anot.isel(time=summer_mask_anot)
    
    # Number of valid time indices
    valid_indices = min(len(anos_summer.time), len(anot_summer.time))

    # Parallel overlap computation
    alphas = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(
        delayed(process_single_time)(i, anos_summer, anot_summer)
        for i in range(valid_indices)
    )
    
    return alphas


# Sort the dataset by latitud
original_ds = anos2.sortby('latitude') 

# Define coarsened longitude and latitude grids
new_lon = original_ds.longitude.values[::3]
new_lat = original_ds.latitude.values[::3]
time = original_ds.time.values

# Initialize an empty array for storing ASC data
sai_data = np.full((len(time), len(new_lat), len(new_lon)), np.nan)

# Create an xarray dataset template for ASC storage
new_ds = xr.Dataset(
    {
        'ASC': (['time', 'latitude', 'longitude'], sai_data)
    },
    coords={
        'time': time,
        'latitude': new_lat,
        'longitude': new_lon
    }
)


# Calculation
for i in range(0,600,3):
    for j in range(0,1440,3):
        anosM=anos2.isel(latitude=slice(i,i+3),longitude=slice(j,j+3))
        anotM=anot2.isel(latitude=slice(i,i+3),longitude=slice(j,j+3))
        if ~np.isnan(anosM).all():
            paratest=process_no_lag_parallel(anosM, anotM, n_jobs=12)
            new_ds['ASC'][:,int(i/3),int(j/3)]=paratest
            del paratest
        else:
            pass
        del anosM, anotM
        print(i, '-', j, end='\r')
