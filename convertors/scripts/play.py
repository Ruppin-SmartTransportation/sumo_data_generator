import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Load the dataset
file_path = "convertors/Datasets/METR-LA.h5"  # Update this path
with h5py.File(file_path, "r") as f:
    timestamps = f["df"]["axis1"][:]  # Assuming axis1 holds timestamps
    print("Raw timestamps:", timestamps[:5])  # Print first few timestamps

    # Ensure timestamps are integers
    timestamps = timestamps.astype(int)
    # Convert from nanoseconds to seconds
    timestamps = timestamps // 1_000_000_000
    
    # Convert to datetime
    timestamps = pd.to_datetime(timestamps, unit="s")
    print("Converted timestamps:", timestamps[:5])  # Check first few dates

    
    columns = f["df"]["axis0"][:]  # Should be axis0, not axis1
    columns = columns.astype(int)
    
    traffic_data = f["df"]["block0_values"][:]  # (num_samples, num_sensors)

# # Now, timestamps should match num_samples, and columns should match num_sensors
# print("Timestamps shape:", timestamps.shape)  # Should be (34272,)
# print("Columns shape:", columns.shape)  # Should be (207,)
# print("Traffic data shape:", traffic_data.shape)  # Should be (34272, 207)

# Create the DataFrame
df = pd.DataFrame(traffic_data, index=timestamps, columns=columns)
# Replace 0 values with NaN to avoid interpolating valid zero-speed scenarios
df.replace(0, np.nan, inplace=True)

# Interpolate missing values (e.g., linear interpolation along the time axis)
df.interpolate(method="linear", inplace=True)

# Fill any remaining NaNs (e.g., forward-fill or backward-fill)
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)
# Display the first few rows
print(df.head())

# Select a few random sensors
selected_sensors = np.random.choice(df.columns, size=5, replace=False)

# Plot the time series for selected sensors
# plt.figure(figsize=(12, 6))
# for sensor in selected_sensors:
#     plt.plot(df.index, df[sensor], label=f"Sensor {sensor}")

# plt.xlabel("Time")
# plt.ylabel("Traffic Value")
# plt.title("Traffic Time Series for Selected Sensors")
# plt.legend()
# plt.xticks(rotation=45)
# plt.show()


# Select a few random sensors
selected_sensors = np.random.choice(df.columns, size=2, replace=False)

# Plot the time series for selected sensors
# plt.figure(figsize=(12, 6))
# for sensor in selected_sensors:
#     plt.plot(df.index[:300], df[sensor][:300], label=f"Sensor {sensor}")

# plt.xlabel("Time")
# plt.ylabel("Traffic Value")
# plt.title("Traffic Time Series for Selected Sensors")
# plt.legend()
# plt.xticks(rotation=45)
# plt.show()


# # Sample a subset of sensors to keep the heatmap readable
# num_sensors = 50  # Adjust this number as needed
# sampled_sensors = np.random.choice(df.columns, size=num_sensors, replace=False)

# # Extract the subset of data
# df_subset = df[sampled_sensors]

# # Plot the heatmap
# plt.figure(figsize=(14, 8))
# sns.heatmap(df_subset.T, cmap="viridis", cbar=True, xticklabels=False)

# plt.xlabel("Time")
# plt.ylabel("Sensor ID")
# plt.title("Traffic Intensity Heatmap (Sampled Sensors)")
# plt.show()


# # Convert index to hours/minutes for better readability
# df_subset = df.iloc[:100, :50]  # Adjust the subset size as needed
# df_subset.index = pd.to_datetime(df_subset.index)

# # Plot heatmap
# plt.figure(figsize=(12, 6))
# sns.heatmap(df_subset.T, cmap="coolwarm", cbar=True)

# plt.xlabel("Time")
# plt.ylabel("Sensor ID")
# plt.title("Traffic Speed Heatmap")

# # Rotate x-axis labels for better visibility
# plt.xticks(rotation=45)
# plt.show()

# Compute average speed across all sensors for each timestamp
df["Avg_Speed"] = df.mean(axis=1)

# Resample to hourly averages
hourly_speed = df["Avg_Speed"].resample("H").mean()

# Plot rush hour trends
plt.figure(figsize=(12, 5))
plt.plot(hourly_speed, color="blue", label="Avg Speed")
plt.axvspan("2012-03-01 07:00:00", "2012-03-01 09:00:00", color="red", alpha=0.3, label="Morning Rush")
plt.axvspan("2012-03-01 16:00:00", "2012-03-01 18:00:00", color="orange", alpha=0.3, label="Evening Rush")
plt.xlabel("Time of Day")
plt.ylabel("Average Speed (mph)")
plt.title("Rush Hour Traffic Analysis")
plt.legend()
plt.show()


# Extract hour from timestamps
df["Hour"] = df.index.hour

# Compute average speed for each hour of the day across all days
hourly_avg_speed = df.groupby("Hour")["Avg_Speed"].mean()

# Plot the average daily traffic trend
plt.figure(figsize=(12, 5))
plt.plot(hourly_avg_speed, marker="o", linestyle="-", color="blue", label="Avg Speed")

# Highlight rush hours
plt.axvspan(7, 9, color="red", alpha=0.3, label="Morning Rush")
plt.axvspan(16, 18, color="orange", alpha=0.3, label="Evening Rush")

# Labels and titles
plt.xlabel("Hour of the Day")
plt.ylabel("Average Speed (mph)")
plt.title("Average Traffic Pattern Throughout the Day")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True)
plt.show()