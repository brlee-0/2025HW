############################################
# Figure 1 # ###############################
# Fig.1(a) #________________________________

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import scipy.stats as st
import statsmodels.api as sm_stats
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
from cmap import Colormap

qq = 1.5
fig = plt.figure(figsize=(18*qq, 6.5*qq))
gs = gridspec.GridSpec(2, 20, height_ratios=[0.9, 1], hspace=0.5)

Y=TYKJM3['mx2t'].values #Yearly HWD values (https://github.com/brlee-0/2025HW/blob/main/HWD_cal.py)
X=TYKJM3['time'].dt.year.values
X_with_const = sm_stats.add_constant(X)
model = sm_stats.OLS(Y, X_with_const).fit() # Regression

ax3 = fig.add_subplot(gs[0, :12])
ax3.plot(X, Y, lw=4, color='#f53864', zorder=7, label='HWD')
top3 = TYKJM3['mx2t'].nlargest(dim='time', n=3)
ax3.scatter( # Top 3 years
    top3['time'].dt.year,
    top3,
    s=500,
    marker='*',
    color='#d91c48',
    edgecolor='k',
    linewidth=1.2,
    zorder=8,
    label='Top 3 years'
) 
ax3.set_yticks(np.linspace(0,25,6))
ax3.set_xticks(np.arange(1980,2026,5))
ax3.grid(ls='--')
ax3.set_ylabel('days', fontsize=18)
ax3.set_title('(a) KJ heatwave days in JJA 2025',
              loc='left',
              fontsize=26)
ax3.plot(X, # Regression line
         X*model.params[1]+model.params[0],
         ls='--',
         lw=3,
         color='gray',
         alpha=0.8,
         zorder=1,
         label=f'y={model.params[1]:.1e}x{model.params[0]:+.1e}'
        )

for ax in [ax3]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_zorder(10)
    ax.tick_params(axis='both', width=2, labelsize=15, zorder=8)

    
# Fig.1(b, c) #_____________________________

# To calculate anomaly and significance values
def anomaly_significance_simple(anomaly_data, target_year):
    climatology_std = anomaly_data.std(dim='time')

    target_anomaly = anomaly_data.sel(
        time=anomaly_data.time.dt.year == target_year
    ).mean(dim='time')
    
    # z-score
    z_score = target_anomaly / climatology_std
    
    # 95% : |z| > 1.96
    significant = np.abs(z_score) > 1.96
    
    return xr.Dataset({
        'anomaly': target_anomaly,
        'z_score': z_score,
        'significant': significant,
        'positive_anomaly': significant & (z_score > 0),
        'negative_anomaly': significant & (z_score < 0),
    })

result = anomaly_significance_simple(AAtm['t2m'], target_year=2025)


# To plot
qq = 1.5
fig = plt.figure(figsize=(13*qq, 6.6*qq),zorder=8)
gs = gridspec.GridSpec(2, 20, height_ratios=[0.9, 1], hspace=0.5)
lonlat=10
ggb=[-180,180,0,80]

ax4 = fig.add_subplot(gs[0, :])
# blkjja : Blocking index (https://github.com/brlee-0/2025HW/blob/main/Blocking_index.py)
ax4.bar(blkjja.time.dt.dayofyear, blkjja, color='orange', alpha=0.5, 
        width=1, label='Blocking index', zorder=1)
ax4.set_xlabel('Day', fontsize=18)
ax4.set_ylabel(r'[m $degree^{-1}$]', fontsize=15, color='#bf7d02')
ax4.tick_params(axis='y', labelcolor='#bf7d02', labelsize=15)
ax4.invert_yaxis()  

jja_starts = [152, 167, 182, 197, 213, 228, 245]
jja_labels = ['Jun 1', 'Jun 15', 'Jul 1', 'Jul 15', 'Aug 1', 'Aug 15', 'Sep 1']
ax4.set_xticks(jja_starts)
ax4.set_xticklabels(jja_labels, fontsize=13)
ax4.set_xlim(150, 245)
ax4.set_ylim(20, 0.0)


ax6 = ax4.twinx()                   
ax6.spines['left'].set_position(('axes', -0.05))
ax6.yaxis.set_label_position('left')
ax6.yaxis.tick_left()
ax6.set_ylim(0,1)
ax6.tick_params(axis='y', labelcolor='#f53864', labelsize=15)
ax6.set_ylabel('Ratio of HW grid [-]', fontsize=15, color='#f53864')
ax6.bar(Precipt.time.dt.dayofyear, ex25, label='HWD', 
         color='#f53864', width=1, alpha=0.9, zorder=1)


ax5 = ax4.twinx()
ax5.plot(tsMKJ.dayofyear, tsMKJ.t2m, label='T2m climatology', 
         c='black', lw=5, alpha=0.8, zorder=8)
ax5.plot(tsKKJ.dayofyear, tsKKJ.t2m, label='T2m', 
         c='#5b08c7', lw=4, zorder=4)
ax5.set_ylabel('[K]', fontsize=18, color='black')
ax5.set_title('(b) 2025 JJA time series', loc='left', fontsize=28)
ax5.tick_params(axis='y', labelcolor='black', labelsize=15)

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax5.get_legend_handles_labels()
lines3, labels3 = ax6.get_legend_handles_labels()
ax5.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3 ,
           fontsize=14, loc='upper left', bbox_to_anchor=(0, 0.85), framealpha=0.5)

ax4.grid(True, alpha=0.3, linestyle='--', zorder=0)


ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '110m', facecolor='none', edgecolor='none')
land = cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='none', edgecolor='#9e927b', linewidth=3)

ax1 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree(central_longitude=80))
ax1.add_feature(ocean)
ax1.set_extent(ggb, crs=ccrs.PlateCarree())
ax1.set_aspect('auto')
ax1.add_feature(land)

# Statistically significant values are marked with dots
varMin, varMax = -2., 2.
data = result['anomaly']
lons = result.longitude.values
lats = result.latitude.values
cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
levels = np.linspace(varMin, varMax, 40)

cnplot1 = ax1.contourf(cyclic_lons, lats, cyclic_data, cmap=cmap_bo,
                       levels=levels, alpha=0.97, zorder=1,
                       transform=ccrs.PlateCarree(), extend='both')

sig_data = result['significant'].values
sig_cyclic, sig_lons = add_cyclic_point(sig_data, coord=lons)

dint = 8
LON, LAT = np.meshgrid(sig_lons, lats)
lon_sig = np.where(sig_cyclic == 1, LON, np.nan)
lat_sig = np.where(sig_cyclic == 1, LAT, np.nan)

ax1.plot(
    lon_sig[::dint, ::dint], 
    lat_sig[::dint, ::dint], 
    '.', 
    color='black', 
    markersize=2, 
    alpha=0.6,
    transform=ccrs.PlateCarree()
)

ax1.set_title('(c) T2m, GPH500 anomaly fields in JJA 2025', loc='left', fontsize=28)


# Z500 contour
DATA = det_z[-1]/9.8
lons=DATA.longitude.values
lats=DATA.latitude.values
cyclic_DATA, cyclic_lons = add_cyclic_point(DATA, coord=lons)
cs = ax1.contour(cyclic_lons, lats, cyclic_DATA, 
                 levels=np.arange(20, 241, 40),
                 transform=ccrs.PlateCarree(), colors='k', linewidths=2, 
                 negative_linestyles='--', zorder=8)
ax1.clabel(cs, inline=True, fontsize=12, fmt='%d')

# colorbar
cbar_ax1 = fig.add_axes([0.911, 0.133, 0.0115, 0.28]) 
cb1 = fig.colorbar(cnplot1, cax=cbar_ax1, orientation='vertical', shrink=0.2, extend='both')
cb1.set_ticks(np.linspace(-2,2,5))
cb1.ax.tick_params(labelsize=15)
cb1.ax.text(2.5, -0.13, r'$[K]$', rotation=90,
           fontsize=18.5, va='bottom', ha='left')
for spine in cb1.ax.spines.values():
    spine.set_linewidth(2)
    

ax1.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(0, 86, 10), crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
ax1.tick_params(axis='both', width=2, labelsize=12, zorder=8)

# Korea-Japan region
proj = ccrs.PlateCarree()
min_lon, max_lon = 123, 146
min_lat, max_lat = 30, 42

rect = Rectangle((min_lon, min_lat), 
                 max_lon - min_lon,
                 max_lat - min_lat,
                 linewidth=6, edgecolor='k', facecolor='none',
                 transform=proj, zorder=7) 
rect2 = Rectangle((min_lon, min_lat), 
                 max_lon - min_lon,
                 max_lat - min_lat,
                 linewidth=3, edgecolor='white', facecolor='none',
                 transform=proj, zorder=7) 

ax1.add_patch(rect)
ax1.add_patch(rect2)

for ax in [ax1, ax4, ax6]:,
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_zorder(10)
    ax.tick_params(axis='both', width=2, labelsize=15, zorder=8)


# Fig.1(d) #________________________________

fig, ax = plt.subplots(figsize=(13, 4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=80)})
uPDOslp=Auwnd2['u'][-1].sel(latitude=slice(90,-0.0)) # Anomaly of 2025 JJA u wind 
p = uPDOslp.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=sei_cmap,
    robust=True,
    add_colorbar=False,
    vmax=7., vmin=-7.,
    extend='both'
)

gphPDO=muwind['u'].sel(latitude=slice(90., -0.0)) # muwind : mean u wind in JJA
contour = ax.contour(
    gphPDO.longitude, gphPDO.latitude, gphPDO,
    transform=ccrs.PlateCarree(),
    colors='black', linewidths=1, levels=[10,20], # Only level 10-20m/s are shown
)
ax.clabel(contour, inline=True, fontsize=12, fmt='%.0f')

# Statistically significant values are marked with dots
lons = gphPDO.longitude.values
lats = gphPDO.latitude.values
sig_data = xr.DataArray(u_pval, coords=anom_2025.coords).sel(latitude=slice(90., -0.01))
sig_cyclic, sig_lons = add_cyclic_point(sig_data.values, coord=lons)

dint = 8
LON, LAT = np.meshgrid(sig_lons, lats)
percenti=0.05
lon_sig = np.where(sig_cyclic < percenti, LON, np.nan)
lat_sig = np.where(sig_cyclic < percenti, LAT, np.nan)

ax.plot(
    lon_sig[::dint, ::dint],
    lat_sig[::dint, ::dint],
    '.', color='black', markersize=1.5, alpha=0.6,
    transform=ccrs.PlateCarree()
)

ax.set_extent([0, 360, -0.01, 80.1], crs=ccrs.PlateCarree())
cbar = plt.colorbar(p, ax=ax, orientation='vertical',
                    shrink=0.515, pad=0.015, aspect= 13, extend='both')
cbar.set_ticks(np.arange(-7,7.01,3.5))
cbar.ax.tick_params(labelsize=12)        
cbar.set_label('$[m/s]$', fontsize=12) 

ax.coastlines(color='#3b3934', linewidth=1.9)
ax.add_feature(cfeature.BORDERS, linewidth=0.8,color='#242321',alpha=0.6)

gl = ax.gridlines(draw_labels=True, ls='--', linewidth=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

# Korea-Japan region
proj = ccrs.PlateCarree()
min_lon, max_lon = 123, 146
min_lat, max_lat = 30, 42

rect = Rectangle((min_lon, min_lat), 
                 max_lon - min_lon,
                 max_lat - min_lat,
                 linewidth=4, edgecolor='k', facecolor='none',
                 transform=proj, zorder=7) 
rect2 = Rectangle((min_lon, min_lat), 
                 max_lon - min_lon,
                 max_lat - min_lat,
                 linewidth=2, edgecolor='white', facecolor='none',
                 transform=proj, zorder=7) 

ax.add_patch(rect)
ax.add_patch(rect2)


for spine in ax.spines.values():
    spine.set_linewidth(2.2)
    spine.set_zorder(10)
ax.tick_params(axis='both', width=2, labelsize=16, zorder=8)

cbar.ax.tick_params(labelsize=13, width=2)
for spine in cbar.ax.spines.values():
    spine.set_linewidth(2.2)

ax.tick_params(axis='both', width=2, labelsize=15, zorder=8)
ax.set_title(f"(d) 200 hPa u wind anomaly and climatological jet in JJA 2025",loc='left', fontsize=19)
plt.tight_layout()
plt.show()

############################################
# Figure 2 # ###############################
# Fig.2(a) #________________________________

fig, ax1 = plt.subplots(figsize=(15, 3.8))
# Jet latitude, Maximum T gradient latitude, PDO index plot
ax1.plot(det_Yjet.time,det_Yjet, linewidth=3.5, color='#000000', label='Jet latitude')
det_dTdy_lat_0_180.plot(ax=ax1, ls='--', alpha=.8, linewidth=2.5, color='#f00553', label='Maximum T gradient latitude')
ax1.set_ylabel('[degree]', fontsize=15)
ax1.set_ylim(-3,3)
ax1.tick_params(axis='y', labelsize=17)
ax1.tick_params(axis='x', labelsize=17)
ax2 = ax1.twinx()
ax2.plot(det_dTdy_lat['time'] , det_PDOidx, ls='--', alpha=.7, linewidth=2.5, color='#005eff', label='PDO index', zorder=3)
ax2.set_ylabel('[-]', fontsize=16, color='#0042b5')
ax2.set_yticks(np.linspace(-3,3,5))
ax2.tick_params(axis='y', labelcolor='#0042b5', labelsize=18)
ax2.set_ylim(-3,3)

# 2025
ax1.plot(
    det_dTdy_lat.time.values[-1], det_dTdy_lat_0_180.values[-1],
    marker='*', markersize=18, color='#d40047', zorder=6
)
ax2.plot(
    det_dTdy_lat.time.values[-1], det_PDOidx[-1],
    marker='*', markersize=18, color='#0042b5', zorder=6
)
ax1.plot(
    det_dTdy_lat.time.values[-1], det_Yjet5[-1],
    marker='*', markersize=18, color='#000000', zorder=6
)
ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_visible(False) 

star_2025 = Line2D(
    [0], [0],
    marker='*',
    color='k',
    linestyle='None',
    markersize=15,
    label='2025'
)


for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.tick_params(width=2.2)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()


text_handle = Patch(color='none')

handles = [h1[0], h1[1], h2[0]]
labels  = [l1[0], l1[1], l2[0]]

leg = ax1.legend(handles, labels, ncol=3, frameon=True,
                 facecolor='white', 
                 #facecolor='none', 
                 edgecolor='none', 
                 framealpha=0.8, 
                 fontsize=16, 
                 loc='lower left',
                 bbox_to_anchor=(0, -.01) 
                )

leg.set_zorder(8)


aa=-0.0
ax1.grid(True, which='major', axis='both', linewidth=1, ls='--')
plt.title('(a) Jet latitude and PDO index', fontsize=23, loc='left')
plt.show()



# Fig.2(b–e) #______________________________

# det_pdoY : detrend PDO Y
# block[['binary_filt_det','magnitude_filt_det']] : detrend blocking index
# det_KJRdwn_sw : detrend SW down
# SST4 : detrend SST
# det_Yjet : detrend jet latitude 

panel_data = [
    (det_pdoY,                            SST4,                                'PDO index [-]',                        'SST [K]'),
    (SST4,                                det_Yjet,                            'SST [K]',                              'Jet latitude [degree]'),
    (det_Yjet,                            block['magnitude_filt_det_mean'],    'Jet latitude [degree]',                r'Blocking index [m $degree^{-1}$]'),
    (block['magnitude_filt_det_mean'],    det_KJRdwn_sw,                       r'Blocking index [m $degree^{-1}$]',    r'SW$_{down}$ [W m$^{-2}$]'),]

titles = ['(b)','(c)','(d)','(e)']
nrows, ncols = 2, 2
fig, axs = plt.subplots(nrows, ncols, figsize=(8.5, 7))
axs = axs.flatten()

for pp, (X, Y, xlabel, ylabel) in enumerate(panel_data):
    ax = axs[pp]

    minX, maxX = X.min(), X.max()
    minY, maxY = Y.min(), Y.max()
    pad_x = (maxX - minX) / 40
    pad_y = (maxY - minY) / 40

    ax.grid(color='lightgray', ls='--', zorder=0, linewidth=1.5)
    ax.axvline(0, color='black', alpha=0.4, linewidth=1.7)
    ax.axhline(0, color='black', alpha=0.4, linewidth=1.7)

    ax.scatter(X, Y, facecolor='#7814b3', edgecolor='#390159', linewidth=1.2, s=80, zorder=2, alpha=0.6)

    years_info = [
        (-1, '2025', '#fc634c', 'r'),
        (-2, '2024', '#f2c324', '#c99e0e'),
        (-3, '2023', '#99d60b', '#5f8701')
    ]
    
    for idx, label, fcolor, ecolor in years_info:
    
        x = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
        y = Y.iloc[idx] if hasattr(Y, 'iloc') else Y[idx]
    
        # white background
        ax.scatter(
            x, y,
            facecolor='white',
            edgecolor='white',
            s=81,
            zorder=3
        )
    
        # filled square
        ax.scatter(
            x, y,
            facecolor=fcolor,
            edgecolor=ecolor,
            s=80,
            zorder=3,
            marker='s',
            alpha=0.8,
            label=label
        )
    
        # outline
        ax.scatter(
            x, y,
            facecolor='none',
            edgecolor=ecolor,
            s=80,
            zorder=3,
            marker='s'
        )    

    # Regression line
    a, b = np.polyfit(X, Y, 1)
    x_fit = np.linspace(minX, maxX, 100)
    ax.plot(x_fit, a * x_fit + b, color='#ff9900', lw=2.2, zorder=3)

    # Correlation coefficient and significant test
    r, pval = st.pearsonr(X, Y)
    sig = '*' if pval < 0.01 else ''
    ax.text(minX - pad_x*4, minY - pad_y*3, f'r={r:.2f}{sig}', fontsize=14)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(minX - pad_x*7, maxX + pad_x*5)
    ax.set_ylim(minY - pad_y*7, maxY + pad_y*7)
    ax.set_title(titles[pp], loc='left', fontsize=14)
    if pp == 0:
        ax.legend(fontsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.7)
    ax.tick_params(axis='both', width=1.7, labelsize=12, zorder=8)
fig.tight_layout()
plt.show()


############################################
# Figure 3 # ###############################
# Fig.3(a–f) #______________________________

colors = ['#2340a8', '#00703c', 'k']
labels = ['2023', '2024', '2025']

fig, axes = plt.subplots(
    2, 3,
    figsize=(18, 9),
    gridspec_kw={'height_ratios': [3, 1]}
)
axes = axes.flatten()
N_TOP = 3
cf = None
dint=6
for i, ax in enumerate(axes):
    is_top = i < N_TOP
    j = i if is_top else i - N_TOP

    if is_top: # Zonal temperature plot (latitude-level)
        data = datas[j]
        cf = ax.contourf(
            data.latitude, data.level, data,
            levels=np.linspace(-1.5,1.5,25),
            cmap=cmap_bo, vmin=-1.5, vmax=1.5, extend='both'
        )
        if sigs[j] is not None:
            sig = sigs[j][:-2, :]
            lat2d, lev2d = np.meshgrid(data.latitude[::dint], data.level)
            sig_sub = sig[:, ::dint]
            ax.scatter(lat2d[sig_sub], lev2d[sig_sub], s=2, c='k')
        ax.set_ylim(1000, 200)
        ax.set_ylabel('Pressure [hPa]' if j == 0 else '', fontsize=20)
        ax.set_xlabel('Latitude' if j == 1 else '', fontsize=20)
        
        ax.set_xlabel('')
        ax.set_yscale('log')
        ax.set_yticks([1000, 850, 700, 600, 500, 400, 300, 200])
        ax.set_yticklabels([1000, 850, 700, 600, 500, 400, 300, 200])
        
    else: # Zonal mean SST plot (latitude-SST) 
        data2 = datas2[j]
        ax.grid(ls='--')
        for k, (d, c) in enumerate(zip(datas2, colors)):
            lw    = 5   if k == j else 3
            alpha = 1.0 if k == j else 0.4
            ax.plot(d.latitude, d.values, linewidth=lw, c=c, alpha=alpha, label=labels[k])

        if j == 0:
            legend_lines = [
                Line2D([0], [0], color=colors[m], lw=4)
                for m in range(3)
            ]
        
            ax.legend(
                legend_lines,
                labels,
                fontsize=14
            )
        
        ax.set_ylim(-0.1, 3)
        ax.set_xlim(0, 90.1)
        ax.set_ylabel('[K]' if j == 0 else '', fontsize=20)
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

    ax.set_title(titles[i], loc='left', fontsize=22)
    ax.tick_params(axis='both', width=2, labelsize=16)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

plt.tight_layout()

# colorbar 
pos = ax.get_position()
cax = fig.add_axes([pos.x1 + 0.02,
                    pos.y0+0.345,
                    0.015,           
                    pos.height+0.25])

cbar = fig.colorbar(cf, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=14, width=1.5)
cbar.set_ticks(np.linspace(-1.5,1.5,7))
cbar.set_label('[K]', fontsize=16)
cbar.outline.set_linewidth(2.5)
plt.show()


############################################
# Figure 4 # ###############################
# Fig.4(a–d) #______________________________

def spatial_significance_test_onesample3(data, target_year, months=[6, 7, 8]):
    summer_data = data.sel(time=data.time.dt.month.isin(months))
    summer_yearly = summer_data.groupby('time.year').mean(dim='time', skipna=True)
    
    climatology = summer_yearly.mean(dim='year', skipna=True)
    climatology_std = summer_yearly.std(dim='year', ddof=1, skipna=True)
    
    target_data = summer_yearly.sel(year=target_year)
    n = len(summer_yearly.year)  
    z_score = xr.where(se > 0, (target_data - climatology) / climatology_std, np.nan)
    
    # two-tailed p-value from normal distribution
    p_value = 2 * (1 - xr.apply_ufunc(
        lambda x: st.norm.cdf(np.abs(x)),
        z_score,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]  
    
    significant = p_value < 0.05
    
    return climatology, target_data, p_value, significant

# Detrended variables
variables = {
    'sst': {
        'data': det_Sst.sel(time=det_Sst.time.dt.month.isin([6,7,8])),
        'title': 'SST Anomaly',
        'cbar_label': r'[K]',
        'cmap': 'RdBu_r',
        'extent': [65, 200, 10, 60]
    },
    
    'Prec': {
        'data': det_tp,
        'title': 'Precipitation Anomaly',
        'cbar_label': r'[mm]',
        'cmap': cmasher_cmap,  
        'extent': [65, 200, 10, 60]  
    },

    'tcc': {
        'data': det_tcc,
        'title': 'Total Cloud Cover Anomaly',
        'cbar_label': r'[-]',
        'cmap': 'BrBG',  
        'extent': [65, 200, 10, 60]
    },
    
    'radi': {
        'data': det_Ready.sel(time=det_Ready.time.dt.month.isin([6,7,8])), 
        'title': 'Net Radiation Anomaly',
        'cbar_label': r'[$W/m^2$]',
        'cmap': bwr_cmap,
        'extent': [65, 200, 10, 60]
    }
}

results = {}
nb=0
for var_name, var_info in variables.items():
    result = spatial_significance_test_onesample3( 
        var_info['data'], target_year=2025, months=[6, 7, 8]
    )
    results[var_name] = result
    nb=nb+1
    print(nb)


cmasher_cmap = Colormap("cmasher:prinsenvlag").to_mpl()
bwr_cmap = Colormap("matplotlib:bwr").to_mpl()
fig = plt.figure(figsize=(20, 9.2))

ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '110m', 
                                     facecolor='none', edgecolor='none')
land = cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                    facecolor='none', edgecolor='k',
                                    linewidth=2)

# threshold factor
threshold_factor = 1  

for idx, (var_name, var_info) in enumerate(variables.items(), 1):
    print(f"  {var_name}...", end=' ')
    
    climatology, mean_target, p_value, significant = results[var_name]
    if isinstance(climatology, xr.Dataset):
        climatology = climatology[list(climatology.data_vars)[0]]
    if isinstance(mean_target, xr.Dataset):
        mean_target = mean_target[list(mean_target.data_vars)[0]]
    if isinstance(p_value, xr.Dataset):
        p_value = p_value[list(p_value.data_vars)[0]]
    if isinstance(significant, xr.Dataset):
        significant = significant[list(significant.data_vars)[0]]
    
    anomaly = mean_target - climatology
    anomaly = anomaly.where(significant)
    
    if var_name == 'Prec':
        anomaly = anomaly * 1000
    
    ax = fig.add_subplot(2, 2, idx, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent(var_info['extent'], crs=ccrs.PlateCarree())

    # vmin/vmax of each variable
    if var_name == 'radi':
        vmn, vmx = -20, 20
    elif var_name == 'tcc':
        vmn, vmx = -0.15, 0.15
    elif var_name == 'sst':
        vmn, vmx = -2, 2
        land = cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                            facecolor='lightgray', edgecolor='k',
                                            linewidth=2)
    elif var_name == 'Prec':
        vmn, vmx = -4, 4
        land = cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                            facecolor='none', edgecolor='k',
                                            linewidth=2)
    
    im = anomaly.plot(
        ax=ax, 
        cmap=var_info['cmap'], 
        center=0,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        vmin=vmn, vmax=vmx
    )

    if var_name == 'Prec':
        uq_mean = YYintds.isel(time=-1)['uq'] # UUintds : Integrated Vapor Transport (IVT) Vector
        vq_mean = YYintds.isel(time=-1)['vq']
        
        z500_data = McomG25['z'] / 9.8
        lon_z500 = MZ.longitude.values
        lat_z500 = MZ.latitude.values
        z500_smooth = gaussian_filter(z500_data.values, sigma=2.0)
        z500_smooth2 = gaussian_filter((McomG['z'] / 9.8).values, sigma=2.0)   

        # Contour of western North Pacific subtropical high (Climatology and 2025 anomaly)
        ax.contour(lon_z500, lat_z500, z500_smooth2,
                   levels=[5880], colors='black', linewidths=9,
                   linestyles='-', transform=ccrs.PlateCarree(), zorder=6) 
        ax.contour(lon_z500, lat_z500, z500_smooth2,
                   levels=[5880], colors='gray', linewidths=4,
                   linestyles='-', transform=ccrs.PlateCarree(), zorder=6)        
        ax.contour(lon_z500, lat_z500, z500_smooth,
                   levels=[5880], colors='black', linewidths=9,
                   linestyles='-', transform=ccrs.PlateCarree(), zorder=6)
        ax.contour(lon_z500, lat_z500, z500_smooth,
                   levels=[5880], colors='#cfff3d', linewidths=4,
                   linestyles='-', transform=ccrs.PlateCarree(), zorder=6)  


        # Contour of Tibetan high (Climatology and 2025 anomaly)
        csuk=ax.contour(
            lon_z500, lat_z500, z200_smooth, levels=[12480],
            colors='#000000', linewidths=10,
            transform=ccrs.PlateCarree())
        labels = ax.clabel(csuk, inline=1, fontsize=14)
        csu = ax.contour(
            lon_z500, lat_z500, z200_smooth, levels=[12480],
            colors='gray', linewidths=5,
            transform=ccrs.PlateCarree())
        
        labels = ax.clabel(csu, inline=1, fontsize=14)
        for label in labels:
            label.set_clip_on(True)

        csu25 = ax.contour(
            lon_z500, lat_z500, z200_smooth25, levels=[12480],
            colors='#000000', linewidths=10,
            transform=ccrs.PlateCarree())
        
        csu25 = ax.contour(
            lon_z500, lat_z500, z200_smooth25, levels=[12480],
            colors='#f53d68', linewidths=5,
            transform=ccrs.PlateCarree())

        # Magnitude filtering  
        skip = 10
        magnitude = np.sqrt(uq_plot**2 + vq_plot**2)
        mean_magnitude = np.nanmean(magnitude)
        threshold = mean_magnitude * threshold_factor

        # masking : if <threshold
        mask = magnitude < threshold
        uq_plot2 = np.where(mask, np.nan, uq_plot)
        vq_plot2 = np.where(mask, np.nan, vq_plot)

        u_masked = uq_plot2[::skip, ::skip]
        v_masked = vq_plot2[::skip, ::skip]
        
        quiver = ax.quiver(Yint_ds.longitude[::skip],
                           Yint_ds.latitude[::skip],
                           u_masked,
                           v_masked,
                           scale=2000, color='black', width=0.003,
                           transform=ccrs.PlateCarree(), zorder=7)
        ax.quiverkey(quiver, 0.83, 1.05, 100, '100 kg/(m·s)',
                     labelpos='E', coordinates='axes', fontproperties={'size': 15})

        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=4, label='Climatology'),
            Line2D([0], [0], color='#cfff3d', linewidth=4, label='5880gpm (500hPa)'),
            Line2D([0], [0], color='#f53d68', linewidth=4, label='12480gpm (200hPa)')
        ]
        legend = ax.legend(handles=legend_elements, fontsize=15, loc='upper left',
                          frameon=True, framealpha=0.9, facecolor='white')
        legend.set_zorder(20)
    
    ax.add_feature(ocean)
    ax.add_feature(land)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.66, pad=0.031, aspect=15, extend='both')
    cbar.set_label(var_info['cbar_label'], fontsize=15)
    cbar.ax.tick_params(labelsize=14, width=2)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xticks(np.arange(65, 201, 20), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 61, 10), crs=ccrs.PlateCarree())
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='both', width=2, labelsize=14)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    panel_label = ['(a)', '(b)', '(c)', '(d)'][idx-1]
    ax.set_title('')
    ax.set_title(f'{panel_label} {var_info["title"]}', 
                 fontsize=29, loc='left', pad=10)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


############################################
# Figure 5 # ###############################
# Fig.5(a,b) #______________________________

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cmap import Colormap
        
guppy_cmap = Colormap("cmasher:guppy_r").to_mpl()
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '110m', 
                                     facecolor='none', edgecolor='none')
land = cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                    facecolor='lightgray', edgecolor='k',
                                    linewidth=2)  


fig, axes = plt.subplots(1, 2, figsize=(17, 6.8),
                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# Climatology of air-sea coupling
ax1 = axes[0]
ax1.set_extent([90, 200, 10, 70], crs=ccrs.PlateCarree())  

im1 = clim_ds['SAI'].plot(ax=ax1, cmap=guppy_cmap, vmin=0.04, vmax=0.12, extend='both',
                     transform=ccrs.PlateCarree(),
                     add_colorbar=False)  
ax1.add_feature(ocean)
ax1.add_feature(land)
ax1.set_xticks(np.arange(90, 201, 20), crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(10, 71, 10), crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.tick_params(axis='both', labelsize=15, width=2)  
ax1.set_title('')
ax1.set_title('(a) Climatology of ASC', fontsize=28, loc='left')
for spine in ax1.spines.values():
    spine.set_linewidth(2.5)

# Colorbar 1
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.50, pad=0.03, extend='both')
cbar1.set_ticks(np.arange(0.04, 0.121, 0.02))
cbar1.ax.tick_params(labelsize=15, width=2.1) 
cbar1.set_label('[-]', fontsize=15)
for spine in cbar1.ax.spines.values():
    spine.set_linewidth(2.1)

                
# 2025 anomaly of air-sea coupling
ax2 = axes[1]
ax2.set_extent([90, 200, 10, 70], crs=ccrs.PlateCarree())  

im2 = ANSD2[-1].plot(ax=ax2, cmap='seismic', vmin=-0.3, vmax=0.3, extend='both',
                     transform=ccrs.PlateCarree(),
                     add_colorbar=False)
ax2.add_feature(ocean)
ax2.add_feature(land)
ax2.set_xticks(np.arange(90, 201, 20), crs=ccrs.PlateCarree())
ax2.set_yticks(np.arange(10, 71, 10), crs=ccrs.PlateCarree())
ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax2.yaxis.set_major_formatter(cticker.LatitudeFormatter())

ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.tick_params(axis='both', labelsize=15, width=2.1)
ax2.set_title('')
ax2.set_title('(b) 2025 anomaly of ASC', fontsize=28, loc='left')
for spine in ax2.spines.values():
    spine.set_linewidth(2.5)

# Colorbar 2
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.50, pad=0.03, extend='both')
cbar2.ax.tick_params(labelsize=15, width=2.1) 
cbar2.set_ticks(np.arange(-0.3, 0.31, 0.15))
cbar2.set_label('[-]', fontsize=15)
for spine in cbar2.ax.spines.values():
    spine.set_linewidth(2.1) 

plt.tight_layout()
plt.show()



############################################
# Figure 5 # ###############################
# Fig.5(c,d) #______________________________

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
    
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 4.5), 
                               gridspec_kw={'width_ratios': [1.4, 1]})

# Time series of air-sea coupling strangth
# T2m
ax1.plot(Yanot.time.dt.year, Yanot, label='T2m', color='orange', linewidth=4)
ax1.set_title('(c) Time series of ASC', loc='left', fontsize=25)
ax1.set_xlabel('Year', fontsize=15)
ax1.set_ylabel('[K]', fontsize=15, color='black')
ax1.tick_params(axis='y', labelcolor='#d16111', labelsize=15, width=2)
ax1.tick_params(axis='x', labelcolor='black', labelsize=15, width=2)
ax1.grid(alpha=0.3)
ax1.set_ylim(-2, 2.5)
ax1.set_yticks(np.linspace(-2, 2., 5))

# ASC
ax2 = ax1.twinx()
ax2.plot(Yanot.time.dt.year, NSD, label='ASC', color='navy', linewidth=4, linestyle='-')
ax2.set_ylabel('[-]', fontsize=15, color='navy')
ax2.tick_params(axis='y', labelcolor='navy', labelsize=15, width=2)
ax2.tick_params(axis='x', labelsize=15, width=2)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, 
           loc='best', fontsize=15, framealpha=0.95, ncol=1)
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# 15 year moving correlation
s1 = pd.Series(Yanot.values)
s2 = pd.Series(NSD)
s3 = pd.Series(Yanos.values)
s4 = pd.Series(TZdsm['alpha'].values)

rolling_corr_t2m = s1.rolling(15, center=True).corr(s2)
rolling_corr_Z = s1.rolling(15, center=True).corr(s4)
rolling_corr_sst = s3.rolling(15, center=True).corr(s2)

time_subset = Yanot.time
ax3.plot(time_subset, rolling_corr_t2m, 
         label='r(T2m,ASC)', color='#426311', linewidth=4)
ax3.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)

ax3.set_title('(d) 15 year moving correlation', loc='left', fontsize=25)
ax3.set_xlabel('Year', fontsize=15)
ax3.set_ylabel('r[-]', fontsize=15)
ax3.set_ylim(-0.6, 1.)
ax3.axhline(0.5,color='gray',ls='--')
ax3.axhline(-0.5,color='gray',ls='--')
ax3.axhspan(0.5,1.5, color='#ffde3b', alpha=0.2)
ax3.axhspan(-1,-0.5, color='#ffde3b', alpha=0.2)
ax3.set_yticks(np.linspace(-0.6, 1., 5))

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.tick_params(axis='both', labelsize=15, width=2)
ax3.legend(loc='best', fontsize=15, framealpha=0.8)
ax3.grid(alpha=0.3)
for spine in ax3.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()

    
############################################
# Figure 6 # ###############################
# Fig.6 bottom SST #________________________

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
import cartopy.io.shapereader as shpreader
    

qq = 1.
fig = plt.figure(figsize=(16*qq, 10*qq))
ax = fig.add_subplot(1, 1, 1)

shpfilename = shpreader.natural_earth(resolution='110m',
                                      category='physical',
                                      name='land')

lon_min, lon_max = 50, 200
lat_min, lat_max = 10, 70

# tilt the map (Affine transformation)
skew_angle = 40
skew = Affine2D().skew_deg(skew_angle, 0)
squash = Affine2D().scale(1.0, 0.52)
trans = squash + skew + ax.transData

# land masking (Polygon patch) 
for record in shpreader.Reader(shpfilename).records():
    geom = record.geometry
    if geom.geom_type == 'Polygon':
        polygons = [geom]
    else:
        polygons = geom.geoms
    
    for polygon in polygons:
        x, y = polygon.exterior.xy
        x = np.array(x)
        y = np.array(y)
        x = np.where(x < 0, x + 360, x)
        
        mask = (x >= lon_min) & (x <= lon_max) & (y >= lat_min) & (y <= lat_max)
        if mask.any():
            coords = np.column_stack((x[mask], y[mask]))
            poly = Polygon(
                coords,
                closed=True,
                facecolor='#a8a59e',
                edgecolor='#a8a59e',
                linewidth=0.6,
                transform=trans,
                zorder=2
            )
            ax.add_patch(poly)

# Coastal line
for record in shpreader.Reader(shpfilename).records():
    geom = record.geometry
    if geom.geom_type == 'Polygon':
        polygons = [geom]
    else:
        polygons = geom.geoms
    
    for polygon in polygons:
        x, y = polygon.exterior.xy
        x = np.array(x)
        y = np.array(y)
        x = np.where(x < 0, x + 360, x)
        
        mask = (x >= lon_min) & (x <= lon_max) & (y >= lat_min) & (y <= lat_max)
        if mask.any():
            ax.plot(x[mask], y[mask],
                   color='#3d3d3d',
                   linewidth=2,
                   transform=trans,
                   zorder=3)

# SST plot
sst_plot = Data_sst2.copy()
lon = Data_sst2.longitude.values
lat = Data_sst2.latitude.values
lon2d, lat2d = np.meshgrid(lon, lat)

mesh = ax.pcolormesh(
    lon2d, lat2d, sst_plot,
    transform=trans,
    shading='auto',
    cmap='seismic',
    zorder=0.3,
    vmax=2, vmin=-2
)

cbar = fig.colorbar(
    mesh,
    ax=ax,
    orientation='horizontal',
    fraction=0.046,
    pad=0.01,
    extend='both'
)
cbar.set_label('SST [K]', fontsize=15)
cbar.set_ticks(np.arange(-2, 2.1, 1))
cbar.ax.tick_params(labelsize=16)

for spine in cbar.ax.spines.values():
    spine.set_linewidth(2.2)
    spine.set_zorder(10)

# Grid
for lon in range(int(lon_min), int(lon_max)+1, 30):
    lats = np.linspace(lat_min, lat_max, 100)
    lons = np.full_like(lats, lon)
    ax.plot(lons, lats, 'gray', linewidth=0.5, alpha=0.5,
            linestyle='--', transform=trans, zorder=1)

for lat in range(int(lat_min), int(lat_max)+1, 10):
    lons = np.linspace(lon_min, lon_max, 100)
    lats = np.full_like(lons, lat)
    ax.plot(lons, lats, 'gray', linewidth=0.5, alpha=0.5,
            linestyle='--', transform=trans, zorder=1)

# Map border
ax.plot([lon_min, lon_max], [lat_min, lat_min], 'k-', 
        linewidth=2, transform=trans, zorder=5)
# Right
ax.plot([lon_max, lon_max], [lat_min, lat_max], 'k-', 
        linewidth=2, transform=trans, zorder=5)
# Upper
ax.plot([lon_max, lon_min], [lat_max, lat_max], 'k-', 
        linewidth=2, transform=trans, zorder=5)
# Left
ax.plot([lon_min, lon_min], [lat_max, lat_min], 'k-', 
        linewidth=2, transform=trans, zorder=5)
tick_length = 2

# Bottom
lon_ticks = [75, 100, 125, 150, 175]  
for lon in lon_ticks:
    ax.plot([lon, lon], [lat_min, lat_min-tick_length], 'k-', 
            linewidth=2, transform=trans, zorder=5, clip_on=False)
    ax.text(lon, lat_min-tick_length-2.5, f'{lon}°E', 
            ha='center', va='top', fontsize=18, transform=trans)

ax.text(lon_min, lat_min-tick_length-2.5, f'{lon_min}°E', 
        ha='center', va='top', fontsize=18, transform=trans)
ax.text(lon_max, lat_min-tick_length-2.5, f'{lon_max}°E', 
        ha='center', va='top', fontsize=18, transform=trans)

lat_ticks = [30, 50]  
for lat in lat_ticks:
    ax.plot([lon_min, lon_min-tick_length], [lat, lat], 'k-', 
            linewidth=2, transform=trans, zorder=5, clip_on=False)
    ax.text(lon_min-tick_length-2.5, lat, f'{lat}°N', 
            ha='right', va='center', fontsize=18, transform=trans)

ax.text(lon_min-tick_length-2.5, lat_min, f'{lat_min}°N', 
        ha='right', va='center', fontsize=18, transform=trans)
ax.text(lon_min-tick_length-2.5, lat_max, f'{lat_max}°N', 
        ha='right', va='center', fontsize=18, transform=trans)

# Tick setting
ax.set_xlim(lon_min - 15, lon_max + 30)
ax.set_ylim(lat_min - 15, lat_max + 15)

ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
