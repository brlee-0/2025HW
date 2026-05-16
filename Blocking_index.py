#### Blocking index ####
# Blocking magnitude calculation method used in the NOAA GFDL TM90 code #


# Module 
import numpy as np
import xarray as xr

from glob import glob
from scipy import ndimage

# JJA data open
Glist=sorted(glob('/ERA5/hourly/lev/*/06/*/geopotential/500.nc')+glob('/ERA5/hourly/lev/*/07/*/geopotential/500.nc')+glob('/ERA5/hourly/lev/*/08/*/geopotential/500.nc'))
gp=xr.open_mfdataset(Glist, chunks={"time": 24})

# Blocking index domain used in this study
gp_sel = gp["z"].sel(
    latitude=slice(65, 15),
    longitude=slice(0, 180)
)
gp_daily = gp_sel.resample(time="1D").mean() # Hourly to daily
z500 = (gp_daily / 9.80665) # Geopotential unit : m

# Set three main centered latitudes, and thresholds 
phi_0 = np.array([35., 40., 45.])
phi_s = phi_0 - 20.   # [ 15, 20, 25 ]
phi_n = phi_0 + 20.   # [ 55, 60, 65 ]
ghgs_thresh = 0.
ghgn_thresh = -10.

# Initialize arrays to save the cumulative blocking magnitude and the number of blocking occurences
block_sum = xr.full_like(
    z500.isel(latitude=0).drop_vars("latitude"),
    fill_value=0., dtype=np.float32
)
block_cnt = xr.full_like(
    z500.isel(latitude=0).drop_vars("latitude"),
    fill_value=0., dtype=np.float32
)

# Calculation (south, main center, north)
for i, (ps, p0, pn) in enumerate(zip(phi_s, phi_0, phi_n)):
    print(i, end='\r')
    Z_s = z500.sel(latitude=ps, method="nearest").drop_vars("latitude")
    Z_0 = z500.sel(latitude=p0, method="nearest").drop_vars("latitude")
    Z_n = z500.sel(latitude=pn, method="nearest").drop_vars("latitude")

    # Selected latitude values
    ps_act = float(z500.latitude.sel(latitude=ps, method="nearest"))
    p0_act = float(z500.latitude.sel(latitude=p0, method="nearest"))
    pn_act = float(z500.latitude.sel(latitude=pn, method="nearest"))
    
    # Calculate geopotential height gradient
    GHGS = (Z_0 - Z_s) / (p0_act - ps_act)
    GHGN = (Z_n - Z_0) / (pn_act - p0_act)
    cond = (GHGS > ghgs_thresh) & (GHGN < ghgn_thresh) # Determine whether the blocking condition is satisfied
    magnitude = ((2*Z_0 - Z_s - Z_n) / (pn_act - ps_act)).where(cond, other=0.)
    
    block_sum = block_sum + magnitude
    block_cnt = block_cnt + cond.astype(np.float32)
    
# Compute the mean blocking magnitude for blocking events
block_mag = (block_sum / block_cnt.where(block_cnt > 0)).fillna(0.)


# Apply the 5-day persistence criterion
block_np = block_mag.values  
block_filtered = np.zeros_like(block_np)

for ilon in range(block_np.shape[1]):
    col = block_np[:, ilon]
    mask = col > 0
    labeled, n_features = ndimage.label(mask)

    for label_id in range(1, n_features + 1):
        idx = np.where(labeled == label_id)[0]

        if len(idx) >= 5:
            block_filtered[idx, ilon] = col[idx]

# Save as an xarray
block_mag_filtered = xr.DataArray(
    block_filtered,
    coords=block_mag.coords,
    dims=block_mag.dims,
    attrs={"long_name": "Blocking magnitude (>=5 consecutive days)"}
)
