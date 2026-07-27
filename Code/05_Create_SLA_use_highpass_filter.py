"""
===============================================================================
Sea Level Anomaly (SLA) High-Pass Filtering Using a Cosine Window
===============================================================================

Description:
    This script applies a cosine-window high-pass filter to hourly Sea Level
    Anomaly (SLA) observations from NOAA tide gauges to isolate short-term
    water level variability associated with storms. A 20-day cosine low-pass
    filter is first applied to remove seasonal and low-frequency variability.
    The high-pass (storm) signal is then obtained by subtracting the low-pass
    component from the original SLA.

    The filtered SLA is standardized by computing sigma values (z-scores),
    allowing storm-related sea level anomalies to be compared consistently
    across multiple tide gauge stations.

Features:
    - Reads hourly SLA data from multiple tide gauge stations
    - Applies a 20-day cosine low-pass filter
    - Computes high-pass (storm surge) residuals
    - Standardizes residuals as sigma (z-score)
    - Saves filtered datasets for each station in Parquet format
    - Generates an example plot showing the original SLA, low-pass signal,
      and high-pass residual

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - matplotlib
    - pyarrow

Input:
    - Storm catalogue (Parquet)
    - Hourly Sea Level Anomaly (SLA) datasets for selected tide gauges
      in Parquet format

Output:
    One Parquet file per tide gauge containing:
        - time
        - Sea Level Anomaly (SLA)
        - Low-pass filtered SLA
        - High-pass filtered SLA
        - Sigma-normalized high-pass SLA
        - Station latitude
        - Station longitude

Author:
    Dr. Biplab Sadhukhan
    March 27, 2025
    https://github.com/biplabsadhukhan

Notes:
    - Update the input file paths and output directory before running
      the script.
    - Replace placeholder paths (e.g., "/path/to/input/" and
      "/path/to/output_directory/") with directories on your local system.
    - The cosine filter window length (default: 20 days) can be modified
      depending on the application.
    - Sigma values are computed independently for each tide gauge station.
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Paths ----------------

input_file = "/home/biplab/coastal_diff_storm_data_1980-2024.parquet"

sla_files = [
    "/home/biplab/sla_file/Eastport_ME.parquet",
    "/home/biplab/sla_file/Portland_ME.parquet",
    "/home/biplab/sla_file/Boston_MA.parquet",
    "/home/biplab/sla_file/Providence_RI.parquet",
    "/home/biplab/sla_file/Newport_RI.parquet",
    "/home/biplab/sla_file/Bridgeport_CT.parquet",
    "/home/biplab/sla_file/Montauk_NY.parquet",
    "/home/biplab/sla_file/The_Battery_NY.parquet",
    "/home/biplab/sla_file/Cape_May_NJ.parquet",
    "/home/biplab/sla_file/Sandy_Hook_NJ.parquet",
    "/home/biplab/sla_file/Lewes_DE.parquet",
    "/home/biplab/sla_file/Kiptopeke_VA.parquet",
    "/home/biplab/sla_file/Beaufort_NC.parquet",
    "/home/biplab/sla_file/Duck_NC.parquet",
    "/home/biplab/sla_file/Wilmington_NC.parquet",
    "/home/biplab/sla_file/Charleston_SC.parquet"
]


# ---------------- Load Storm Data ----------------
print("Loading storm data...")
df = pd.read_parquet(input_file)
df["time"] = pd.to_datetime(df["time"])

# Filter years
start_year, end_year = 1980, 2024
df = df[(df["time"].dt.year >= start_year) & (df["time"].dt.year <= end_year)]

# ---------------- Load SLA Data ----------------
print("Loading SLA data...")
dfs = []

for file in sla_files:
    sla_df = pd.read_parquet(
        file,
        columns=["time", "sla", "sla_lat", "sla_lon"])
    
    sla_df["time"] = pd.to_datetime(sla_df["time"])
    sla_df["station"] = file.split("/")[-1].replace(".parquet", "")
    
    dfs.append(sla_df)

all_sla_data = pd.concat(dfs, ignore_index=True)

print(f"Total SLA records: {len(all_sla_data)}")

# ---------------- Cosine Weights ----------------
def cosine_weights(window):
    i = np.arange(window)
    
    # Centered cosine window
    weights = np.cos(np.pi * (i - window // 2) / window)
    
    # Remove negative values (tapered cosine)
    weights = np.clip(weights, 0, None)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    return weights

# ---------------- High-Pass Filter ----------------
def high_pass_cosine(df, window_days=20, freq_per_day=24):
    df = df.sort_values("time")
    
    window = window_days * freq_per_day
    weights = cosine_weights(window)
    
    sla = df["sla"].values
    
    # Apply convolution for smoothing (low-pass)
    lowpass = np.convolve(sla, weights, mode='same')
    
    df["sla_lowpass"] = lowpass
    df["sla_highpass"] = df["sla"] - df["sla_lowpass"]
    
    mean_hp = df["sla_highpass"].mean(skipna=True)
    std_hp  = df["sla_highpass"].std(skipna=True)
    
    df["sigma"] = (df["sla_highpass"] - mean_hp) / std_hp
    
    return df

# ---------------- Apply Filter per Station ----------------
print("Applying cosine high-pass filter...")
all_sla_filtered = all_sla_data.groupby("station", group_keys=False).apply(
    lambda x: high_pass_cosine(x, window_days=20, freq_per_day=24)
)

print("Filtering complete!")

# ---------------- Save to Parquet ----------------
output_dir = "/home/biplab/sla_highpass/"

import os
os.makedirs(output_dir, exist_ok=True)

for station, group in all_sla_filtered.groupby("station"):
    
    # Optional: keep only useful columns
    save_df = group[["time", "sla", "sla_lowpass", "sla_highpass",
        "sigma", "sla_lat", "sla_lon"]]
    
    output_path = os.path.join(output_dir, f"{station}_highpass.parquet")
    
    save_df.to_parquet(output_path, compression="snappy", index=False)
    
    print(f"Saved: {output_path}")

# ---------------- Plot Example ----------------
print("Plotting example station...")
station_name = "Newport_RI"

test_df = all_sla_filtered[all_sla_filtered["station"] == station_name]

plt.figure(figsize=(12, 5))
plt.plot(test_df["time"], test_df["sla"], label="SLA", alpha=0.5)
plt.plot(test_df["time"], test_df["sla_lowpass"], label="Low-pass (20-day cosine)", linewidth=2, zorder=100)
plt.plot(test_df["time"], test_df["sla_highpass"], label="High-pass (storm signal)", linewidth=1)

plt.legend(fontsize=16)
plt.title(f"SLA Filtering - {station_name}", fontweight='bold', fontsize=18)
plt.xlabel("Time", fontweight='bold', fontsize=18)
plt.ylabel("Sea Level Anomaly", fontweight='bold', fontsize=18)
plt.grid()

ax = plt.gca()

# Spine thickness
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Bold ticks
ax.tick_params(axis='both', which='major', labelsize=15)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.show()




