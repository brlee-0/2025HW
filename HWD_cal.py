### Heatwave days cal ###

# Module load
import xarray as xr
from dask.distributed import Client

# Data upload
DT=xr.open_dataset('/Data/ERA5/daily/daily_mx2t.nc') # 2m maxmum temperature
mask=xr.open_dataset('/Data/ERA5/Land-Sea_mask.nc') # Land sea mask
JDT=DT.sel(time=DT.time.dt.month.isin([6,7,8])).where(mask) # Land sea mask

# To calculate the 90th threshold
client = Client(n_workers=8) # CPU 

JDT['monthday'] = JDT.time.dt.strftime('%m-%d')
JDT2_lazy = JDT.mx2t.groupby(JDT.monthday).quantile(0.9, dim='time') # Calendar day 90th threshold
JDT2 = JDT2_lazy.compute()
client.close()

# To calculate the heatwave days based on the threshold
DT.coords['monthday'] = DT.time.dt.strftime('%m-%d')
is_exceeded = DT.mx2t.groupby('monthday') > JDT2.mx2t
is_exceeded2=is_exceeded.drop({'monthday','quantile'})

def remove_singletons3(mask): # criterion of 3 consecutive days
    nxt1 = mask.shift(time=-1, fill_value=False)
    nxt2 = mask.shift(time=-2, fill_value=False)
    starts = mask & nxt1 & nxt2
    middles = starts.shift(time=1, fill_value=False)
    ends = starts.shift(time=2, fill_value=False)

    return starts | middles | ends

is_exceeded3=remove_singletons3(is_exceeded2)
td33=is_exceeded3.where(mask) # land sea mask

KJ3=td33.sel(latitude=slice(30,42), longitude=slice(123,146)) # Korea-Japan region
HWD=(KJ3.mean({'latitude','longitude'})).resample(time='YS').sum()
