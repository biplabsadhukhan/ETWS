"""
===============================================================================
Quadrant-Based Regeression Analysis for response of storm surge on ETWS characterstics
===============================================================================

Description:
    This script investigates the influence of extratropical storms
    characteristics on extreme storm surge events recorded at selected NOAA
    tide-gauge stations along the U.S. East Coast. Extreme storm surge events
    (≥3σ above the long-term mean) are identified, and the corresponding storm
    tracks are extracted for statistical analysis.


    Predictor variables are standardized prior to regression, allowing direct
    comparison of regression coefficients and identification of the dominant
    physical controls on storm surge generation.

Features:
    - Identifies extreme storm surge events (≥3σ threshold)
    - Standardizes predictor variables using z-score normalization
    - Performs quadrant-specific Ordinary Least Squares (OLS) regression
    - Computes regression coefficients, p-values, R², and AIC
    - Produces publication-quality figures comparing regression coefficients
      across stations and coastal quadrants

Predictor Variables:
    - Distance from storm center to tide gauge
    - Storm radius
    - Storm translation speed
    - Storm intensity
    - Pseudo wind stress

Response Variable:
    - Maximum storm surge (Sea Level Anomaly)

Methods:
    - Geodesic distance and bearing calculations
    - Coastline geometry analysis
    - Quadrant-based storm classification
    - Z-score standardization
    - Ordinary Least Squares (OLS) regression
    - Statistical significance testing

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - xarray
    - matplotlib
    - cartopy
    - shapely
    - geopy
    - statsmodels

Input:
    - Storm catalogue containing storm-track information
    - Storm surge dataset for multiple NOAA tide-gauge stations
    - Coastline coordinates defining the U.S. East Coast

Output:
    - Quadrant-wise regression models
    - Standardized regression coefficients
    - Model statistics (R², AIC, p-values)
    - Publication-quality comparison figures
    - Summary tables of regression coefficients and significance levels

Author:
    Dr. Biplab Sadhukhan
    September 04 2025
    https://github.com/biplabsadhukhan

Notes:
    - Update all input and output file paths before running the script.
    - Tide-gauge stations, study period, and storm selection threshold can be
      modified for other coastal regions.
    - The regression framework is modular and can be extended by including
      additional meteorological or oceanographic predictors.

===============================================================================
"""
import pandas as pd
import xarray as xr
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
from shapely.geometry import Polygon
import matplotlib as mpl
import os
import statsmodels.formula.api as smf
def compute_bearing(lat1, lon1, lat2, lon2):
  
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon_rad = np.radians(lon2 - lon1)

    x = np.sin(dlon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360  # Normalize to [0, 360)

def assign_quadrant(storm_angle, normal_angle):
    """
    storm_angle : angle from tide gauge to storm center (0–360)
    normal_angle: coastal normal (0–360)
    """

    # Shift angle so normal becomes 0°
    shifted = (storm_angle - normal_angle) % 360

    if 0 <= shifted < 90:
        return "Q1"
    elif 90 <= shifted < 180:
        return "Q2"
    elif 180 <= shifted < 270:
        return "Q3"
    else:
        return "Q4"

########################################################################################
# ---------------- Paths ----------------
input_file = "/ExtraTropicalWinterStorms/Data/stress_distance_all_tidegauges_1980-2024.parquet"

df = pd.read_parquet(input_file)
###############################################################################
stations = {
    #"Eastport_ME": (44.9, -67.0),
    "Portland_ME": (43.66, -70.25),
    #"Boston_MA": (42.36, -71.05),
    "Newport_RI": (41.49, -71.32),
    "The_Battery_NY":(41.05,-71.96),
    #"Cape_May_NJ": (38.97, -74.96),
    #"Lewes_DE":(38.78,-75.12),
    "Beaufort_NC": (34.72, -76.67),
}

############################################
coast_lon = [-80.28, -80.55, -80.54, -80.88, -81.25, -81.43, -81.13, -80.49, -80.01, -79.74, 
             -79.37, -79.21, -78.71, -77.97, -77.73, -77.26, -76.53, -76.06, -75.52, -75.45,
             -75.79, -75.98, -75.89, -75.34, -75.05, -75.07, -74.86, -74.37, -74.11, -73.97, 
             -73.73, -72.85, -71.87, -70.66, -69.93, -70.06, -70.71, -70.60,
             -70.62, -69.57, -68.81, -67.91, -67.17, -66.12, -65.62, -64.84, -64.27, -64.04,
             -63.63, -62.82, -62.54, -61.98, -61.36, -60.97, -60.73, -60.41, -60.08, -59.81]


coast_lat = [27.48, 28.08, 28.48, 29.04, 29.82, 30.85, 31.62, 32.32, 32.61, 32.80, 
             33.02, 33.18, 33.80, 33.85, 34.28, 34.59, 34.60, 35.04, 35.25, 35.63, 
             36.33, 36.91, 37.12, 37.88, 38.40, 38.77, 38.96, 39.39, 39.73, 40.37, 
             40.58, 40.73, 41.06, 41.52, 41.68, 42.07, 42.18, 42.64,
             43.14, 43.81, 44.02, 44.41, 44.68, 43.75, 43.41, 43.83, 44.26, 44.49,
             44.44, 44.72, 44.80, 44.98, 45.16, 45.27, 45.56, 45.63, 45.79, 45.93]

ref_lat=[31.62]

coast_lon = np.array(coast_lon)
coast_lat = np.array(coast_lat)
valid_indices = np.where(np.array(coast_lat) >= ref_lat)[0]
coast_lon1 = coast_lon[valid_indices]
coast_lat1 = coast_lat[valid_indices]

# Compute arc length l from gauge
l = [0.0]
for i in range(1, len(coast_lon1)):
    dist = geodesic((coast_lat1[i-1], coast_lon1[i-1]), (coast_lat1[i], coast_lon1[i])).km
    l.append(l[-1] + dist)
l = np.array(l)
# Reverse the coast path from end → tide gauge
#coast_lat1_rev = coast_lat1[::-1]
#coast_lon1_rev = coast_lon1[::-1]
coast_lat1_rev = coast_lat1
coast_lon1_rev = coast_lon1

coast_bearings = []
for i in range(len(coast_lat1_rev) - 1):
    lat1, lon1 = coast_lat1_rev[i], coast_lon1_rev[i]
    lat2, lon2 = coast_lat1_rev[i + 1], coast_lon1_rev[i + 1]
    bearing = compute_bearing(lat1, lon1, lat2, lon2)
    coast_bearings.append(bearing)
coast_bearings.append(coast_bearings[-1])  # Repeat last to match array length
#coast_bearing_dir = coast_bearings[::-1]
##################################################################################
plt.close()
# ================= YEAR RANGE ====================

start_year, end_year = 1980, 2024
df['time'] = pd.to_datetime(df['time'].values)
df_year = df[(df['time'].dt.year >= start_year) & (df['time'].dt.year <= end_year)]


print(f"Filtering storms from {start_year} to {end_year}")

# Columns that are not SLA/sigma (storm metadata)
storm_meta_cols = [col for col in df.columns if not col.startswith("sla_") and not col.startswith("sigma_")]
storm_tracks_by_station = {}
storm_counts = {}

# ---------------- Extract storms ≥ +1σ ----------------
max_sla_by_station = {}
station_normals = {}
for station in stations:

    
    sigma_col = f"sigma_{station}"
    sla_col = f"sla_{station}"
    dist_col = f"distance_{station}"
    direct_col = f"direction_{station}"
    compass_col = f"compass_{station}"
    scaled_col = f"sla_scaled_{station}"
    
    extreme_storm_ids = df.loc[df[sigma_col] >= 3, "id_year"].unique()

    # Extract full tracks for those storms
    station_tracks = df[df["id_year"].isin(extreme_storm_ids)].copy()
    station_tracks = station_tracks[storm_meta_cols + [sla_col, sigma_col,scaled_col]]
    storm_tracks_by_station[station] = station_tracks
    storm_counts[station] = len(extreme_storm_ids)

    df_station = storm_tracks_by_station[station].copy()
    # Index of max sigma per storm
    idx = df_station.groupby("id_year")[sigma_col].idxmax()
    # Select the max rows, including distance + direction + compass
    df_max = df_station.loc[idx, [
       "id_year", "time", sla_col, sigma_col, dist_col, direct_col, compass_col,
       "stress", "storm_speed_ms", "radius", "intensity", scaled_col, "lat", "lon","central_pressure"]]
    
    df_max = df_max.rename(columns={sigma_col: "max_sigma",scaled_col: "sla_scaled"})
    df_max = df_max[df_max["max_sigma"] > 0]
    max_sla_by_station[station] = df_max.reset_index(drop=True)
    
    tg_lat, tg_lon = stations[station]

    # ======= compute tangent line midpoints and normal =======
    mid_lats = (coast_lat1_rev[:-1] + coast_lat1_rev[1:]) / 2
    mid_lons = (coast_lon1_rev[:-1] + coast_lon1_rev[1:]) / 2

    mid_dists = [geodesic((tg_lat, tg_lon), (mlat, mlon)).km
                 for mlat, mlon in zip(mid_lats, mid_lons)]
    mid_idx = np.argmin(mid_dists)

    latA, lonA = coast_lat1_rev[mid_idx],     coast_lon1_rev[mid_idx]
    latB, lonB = coast_lat1_rev[mid_idx + 1], coast_lon1_rev[mid_idx + 1]

    tangent_bearing = compute_bearing(latA, lonA, latB, lonB)
    normal_angle = (tangent_bearing + 90) % 360
    station_normals[station] = normal_angle 
    print(f"{station}: normal = {normal_angle:.2f}°")
    # -------- Print quadrant global angle ranges --------
    print(f"\nAngle Ranges for Station: {station}")
    print(f"Normal angle = {normal_angle:.2f}°")

    Q1_min = (normal_angle + 0) % 360
    Q1_max = (normal_angle + 90) % 360

    Q2_min = (normal_angle + 90) % 360
    Q2_max = (normal_angle + 180) % 360

    Q3_min = (normal_angle + 180) % 360
    Q3_max = (normal_angle + 270) % 360

    Q4_min = (normal_angle + 270) % 360
    Q4_max = (normal_angle + 360) % 360

    print(f"  Q1: {Q1_min:.2f}°  to  {Q1_max:.2f}°")
    print(f"  Q2: {Q2_min:.2f}°  to  {Q2_max:.2f}°")
    print(f"  Q3: {Q3_min:.2f}°  to  {Q3_max:.2f}°")
    print(f"  Q4: {Q4_min:.2f}°  to  {Q4_max:.2f}°\n")


    
    # ---------------- Assign Quadrants to the max-sigma storm positions ----------------

    # compute bearing from tide gauge -> storm center (vectorized)
    df_max = df_max.copy()
    df_max['bearing_deg'] = compute_bearing(tg_lat, tg_lon, df_max['lat'].values, df_max['lon'].values)

    # assign quadrant using your function (works row-wise)
    df_max['quadrant'] = df_max['bearing_deg'].apply(lambda a: assign_quadrant(a, normal_angle))

    # store back
    max_sla_by_station[station] = df_max

    # ---------- quick diagnostics ----------
    counts = df_max['quadrant'].value_counts().reindex(['Q1','Q2','Q3','Q4']).fillna(0).astype(int)
    #print(f"{station} quadrant counts:\n{counts}\n")
    
results_list = []
plt.close()
# ---------------- PLOTTING ----------------
# Colors per quadrant
quad_colors = {
"Q1": "orange",
"Q2": "brown",
"Q3": "lightblue",
"Q4": "lightgreen"}

def make_sector_polygon(lon_center, lat_center, az1, az2, radius_km, npoints=180):
    """
    Return a shapely Polygon approximating the sector from az1->az2 (degrees)
    out to radius_km (kilometers) using geodesic points.
    az1, az2 in degrees (0=N, increasing clockwise). If az2 < az1 assume wrap.
    """
    # handle wrap: create azimuths that move monotically from az1 to az2 (possibly >360)
    if az2 < az1:
        azs = np.linspace(az1, az2 + 360, npoints)
    else:
        azs = np.linspace(az1, az2, npoints)

    geod = Geodesic()
    outer_pts = []
    for az in azs:
        # Geodesic.direct returns an array of (lon, lat) pairs when points is [(lon, lat)]
        # We pass a single point and extract the first (lon, lat)
        out = geod.direct(points=[(lon_center, lat_center)], azimuths=[az], distances=[radius_km * 1000])[0]
        # out is like [lon, lat, ...] depending on cartopy geodesic version; handle both possibilities
        if len(out) >= 2:
            lon_pt, lat_pt = out[0], out[1]
        else:
            # fallback: treat as tuple
            lon_pt, lat_pt = out
        outer_pts.append((lon_pt, lat_pt))

    # ensure polygon valid: center -> outer ring -> center
    poly_pts = [(lon_center, lat_center)] + outer_pts + [(lon_center, lat_center)]
    return Polygon(poly_pts)

for station in stations:
    print(f"\n=== PLOTTING: {station} ===")
    lat, lon = stations[station]

    df_station_all = max_sla_by_station[station].copy()
    

    # Use station dataframe for sizes/colors
    sla_vals = df_station_all[f"sla_{station}"].copy()
    sla_min, sla_max = sla_vals.min(), sla_vals.max()
    normal_angle = station_normals[station]

    # Quadrant boundaries (az1, az2)
    Q1 = (normal_angle + 0)   % 360, (normal_angle + 90)  % 360
    Q2 = (normal_angle + 90)  % 360, (normal_angle + 180) % 360
    Q3 = (normal_angle + 180) % 360, (normal_angle + 270) % 360
    Q4 = (normal_angle + 270) % 360, (normal_angle + 360) % 360

    quad_bounds = {
        "Q1": Q1,
        "Q2": Q2,
        "Q3": Q3,
        "Q4": Q4
    }
    for qname, (az1, az2) in quad_bounds.items():
        # determine whether this quadrant wraps across 360->0
        wrap = az2 < az1

        dist_col   = f"distance_{station}"
        sla_col    = f"sla_{station}"
        scaled_col = "sla_scaled"
        stress_col = "stress"
        cp_col     = "central_pressure"
        

        df_station = max_sla_by_station[station].copy()

        # Make sure we filter by the computed bearing (bearing_deg) column, not a string column name
        if "bearing_deg" not in df_station.columns:
            # if you actually have a column named direction_<station> in df_station, you can map it; otherwise use bearing_deg
            df_station['bearing_deg'] = compute_bearing(lat, lon, df_station['lat'].values, df_station['lon'].values)

        if wrap:
           
            df_range = df_station[(df_station['bearing_deg'] >= az1) | (df_station['bearing_deg'] <= az2)].copy()
        else:
            df_range = df_station[(df_station['bearing_deg'] >= az1) & (df_station['bearing_deg'] <= az2)].copy()

        # select and rename columns
        df_range = df_range[["id_year","time","radius","intensity",dist_col,
                             "storm_speed_ms",stress_col,cp_col,
                             "lat","lon",scaled_col,sla_col]].dropna()
        df_range = df_range.rename(columns={dist_col: "distance"})


        # Standardize variables (avoid division by zero)
        vars_to_scale = ["distance", "storm_speed_ms", "radius","intensity", stress_col, cp_col]
        for var in vars_to_scale:
            if df_range[var].std(ddof=0) == 0 or np.isnan(df_range[var].std()):
                df_range[f"z_{var}"] = 0.0
            else:
                df_range[f"z_{var}"] = (df_range[var] - df_range[var].mean()) / df_range[var].std()

        # Model
        formula = f"{sla_col} ~ z_distance + z_radius + z_storm_speed_ms + z_intensity+ z_{stress_col}"
        model = smf.ols(formula=formula, data=df_range).fit()
        df_range["predicted"] = model.predict(df_range)

        # append results
        results_list.append({
            "Quadrant": qname,
            "QDirect": (az1, az2),
            "Station": station,
            "AIC": round(model.aic, 2),
            "R2": round(model.rsquared, 2),
            "Samples": len(df_range),
            "coefficients": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict()
        })

        print(f"\n===== {station} — {qname} =====")
        print(model.summary())
        
results_df = pd.DataFrame(results_list)
# ---------- Convert model coefficients & p-values to columns ----------
# Expand the nested dict columns
coeff_df = pd.json_normalize(results_df['coefficients'])
pval_df  = pd.json_normalize(results_df['pvalues'])

# Rename columns to avoid conflicts
coeff_df = coeff_df.add_prefix("coef_")
pval_df = pval_df.add_prefix("p_")
coeff_df = coeff_df.round(2)
pval_df  = pval_df.round(2)
# Concatenate to main table
results_full = pd.concat([results_df.drop(['coefficients','pvalues'], axis=1),
                          coeff_df, pval_df], axis=1)

df_plot = results_full[[

   "Quadrant", "Station",
    "coef_z_distance", "coef_z_radius", "coef_z_storm_speed_ms", "coef_z_intensity","coef_z_stress",
    "p_z_distance", "p_z_radius", "p_z_storm_speed_ms", "p_z_intensity","p_z_stress",]]

# ---------------- GRID FIGURE: each column = quadrant, each row = station ----------------
variables = [
    ("coef_z_distance", "p_z_distance", "Distance"),
    ("coef_z_radius", "p_z_radius", "Radius"),
    ("coef_z_storm_speed_ms", "p_z_storm_speed_ms", "Speed"), 
    ("coef_z_intensity", "p_z_intensity", "Pressure\n Change"),# wrapped
    ("coef_z_stress", "p_z_stress", "Pseudo\nStress")
]

###############################################################
#Change the Quadrants
quadrants = ["Q1", "Q4"]
#quadrants = ["Q2", "Q3"]
stations_list = list(stations.keys())
quad_colors = {
    "Q1": "orange",
    "Q2": "brown",
    "Q3": "lightblue",
    "Q4": "lightgreen"
}

coef_cols = [
    "coef_z_distance",
    "coef_z_radius",
    "coef_z_storm_speed_ms",
    "coef_z_intensity",
    "coef_z_stress"
]

# collect all coefficients across stations & quadrants used
all_coefs = df_plot[(df_plot["Quadrant"].isin(quadrants))
][coef_cols].values.flatten()

# remove NaNs
all_coefs = all_coefs[~np.isnan(all_coefs)]

# symmetric limits around zero (recommended for regression coefficients)
ymax = np.max(np.abs(all_coefs))
ylim = (-0.6 * ymax, 1.1 * ymax) #for q14
#ylim = (-1 * ymax, 1.1 * ymax) #for q23
tick_step = 0.1
yticks = np.arange(
    np.floor(ylim[0] / tick_step) * tick_step,
    np.ceil(ylim[1] / tick_step) * tick_step + tick_step,
    tick_step
)
n_rows = len(stations_list)
n_cols = len(quadrants)

#---------------------------Plot the figures------------------------------#
fig, axes = plt.subplots(
    len(stations), len(quadrants),
    figsize=(8, 10))
for i, station in enumerate(stations_list):
    df_st = df_plot[df_plot["Station"] == station]

    for j, quad in enumerate(quadrants):
        ax = axes[i, j] if n_rows > 1 else axes[j]

        df_q = df_st[df_st["Quadrant"] == quad]

        if df_q.empty:
            ax.set_visible(False)
            continue

        coef_vals = []
        p_vals = []
        for coef_col, p_col, _ in variables:
            coef_vals.append(df_q[coef_col].values[0])
            p_vals.append(df_q[p_col].values[0])

        x = np.arange(len(variables))
        bars = ax.bar(x, coef_vals, color=quad_colors[quad], width=0.6)
       


        # ---- significance markers ----
        ymin, ymax = ax.get_ylim()
        offset = 0.04 * (ymax - ymin)
        for xx, yy, p in zip(x, coef_vals, p_vals):
            if p < 0.07:
                marker, color, ms = "*", "black", 14
            elif p < 0.12:
                marker, color, ms = "o", "magenta", 11
            else:
                marker, color, ms = "X", "red", 11
            ax.plot(xx, yy + np.sign(yy)*offset, marker=marker, color=color, markersize=ms, linestyle="None")

        ax.set_xticks(x)

        # show x-ticks ONLY on bottom row
        if i == n_rows - 1:
            ax.set_xticklabels(
                [v[2] for v in variables],
                fontsize=11, fontweight="bold", rotation=45)
        else:
            ax.set_xticklabels([])
            
        ax.grid(axis="y", linestyle="--", alpha=0.4)   
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        # remove y-ticks for Q4 panels
        if quad == "Q4":
            ax.tick_params(axis='y', which='both',
                   left=False, labelleft=False)

        ax.axhline(0, color="black", linewidth=1)
        #
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

        if j == 0:
            ax.set_ylabel("Coefficient\n(Normalized)", fontsize=12, fontweight="bold")
        if i == 0:
            ax.set_title(quad, fontsize=16, fontweight="bold")
        
        # =========================================================
        # Add significance legend ONLY to bottom-right panel
        # =========================================================
        if (i == n_rows - 1) and (j == n_cols - 1):
        #if (i == n_rows-1) and (j == 0):  
            x0, y0 = 0.56, 0.08   # box anchor (axes coords)
            dy = 0.065

            # Invisible box background
            ax.text(x0, y0, "",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(facecolor="white", edgecolor="black",
                              boxstyle="round,pad=0.5"))

            # Line 1: black star
            ax.plot(x0 - 0.03, y0 + 2*dy, "*",
                    color="black", markersize=6,
                    transform=ax.transAxes)
            ax.text(x0, y0 + 2*dy, " Significant (p < 0.05)",
                    transform=ax.transAxes,
                    fontsize=7, fontweight="bold",
                    verticalalignment="center",
                    horizontalalignment="left")

            # Line 2: magenta circle
            ax.plot(x0 - 0.03, y0 + dy, "o",
                    markerfacecolor="magenta",
                    markeredgecolor="blue",
                    markersize=5,
                    transform=ax.transAxes)
            ax.text(x0, y0 + dy, " Significant (p < 0.1)",
                    transform=ax.transAxes,
                    fontsize=7, fontweight="bold",
                    verticalalignment="center",
                    horizontalalignment="left")

            # Line 3: red X
            ax.plot(x0 - 0.03, y0, "X",
                    color="red", markersize=5,
                    transform=ax.transAxes)
            ax.text(x0, y0, " Not Significant",
                    transform=ax.transAxes,
                    fontsize=7, fontweight="bold",
                    verticalalignment="center",
                    horizontalalignment="left")


# =========================================================
# STATION NAMES ON THE RIGHT SIDE (PER ROW, PROPERLY ALIGNED)
# =========================================================

for row, station in enumerate(stations):

    ax_right = axes[row, -1]

    ax_right.text(
        1.05, 0.5,                        
        station.replace("_", " "),
        transform=ax_right.transAxes,rotation=-90,fontsize=16,
        fontweight="bold",va="center",ha="left")
    
plt.tight_layout()
plt.show()

# ---- save full grid ----
output_file = "/ExtraTropicalWinterStorms/Data/all_stations_quadrants_q14.png"
fig.savefig(output_file, dpi=300, bbox_inches="tight")

