import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Load the dataset

def load_and_normalize_data(filepath):

    with h5py.File(filepath, "r") as f:
        timestamps = f["df"]["axis1"][:]  # Assuming axis1 holds timestamps

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

    # Create the DataFrame
    df = pd.DataFrame(traffic_data, index=timestamps, columns=columns)
    # Replace 0 values with NaN to avoid interpolating valid zero-speed scenarios
    df.replace(0, np.nan, inplace=True)

    # Interpolate missing values (e.g., linear interpolation along the time axis)
    df.interpolate(method="linear", inplace=True)

    # Fill any remaining NaNs (e.g., forward-fill or backward-fill)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    return df_scaled, scaler  # Return scaler for inverse transform later


def create_sliding_window(data, lookback=12, predict_forward=1):
    """
    Converts dataset into sliding window format.

    Args:
        data (pd.DataFrame): Input dataset.
        lookback (int): Number of past time steps to use.
        predict_forward (int): Number of future steps to predict.

    Returns:
        X (np.array): Features.
        Y (np.array): Targets.
    """
    X, Y = [], []
    for i in range(len(data) - lookback - predict_forward):
        X.append(data.iloc[i : i + lookback].values)
        Y.append(data.iloc[i + lookback : i + lookback + predict_forward].values)

    return np.array(X), np.array(Y)

def train_linear_regression(X, Y, test_size=0.2):
    """
    Trains a Linear Regression model on traffic data.

    Args:
        X (np.array): Input features (3D).
        Y (np.array): Target labels (3D).
        test_size (float): Fraction of data for testing.

    Returns:
        model: Trained Linear Regression model.
        X_test (np.array): Test input data.
        Y_test (np.array): True test labels.
        Y_pred (np.array): Model predictions.
    """
    # Reshape 3D input to 2D for Linear Regression
    X_flat = X.reshape(X.shape[0], -1)
    Y_flat = Y.reshape(Y.shape[0], -1)

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_flat, Y_flat, test_size=test_size, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Make predictions
    Y_pred = model.predict(X_test)

    return model, X_test, Y_test, Y_pred

def evaluate_model(Y_test, Y_pred):
    """
    Evaluates model performance.

    Args:
        Y_test (np.array): True labels.
        Y_pred (np.array): Predicted labels.

    Returns:
        dict: MSE and R² scores.
    """
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    return {"MSE": mse, "R2": r2}

def plot_predictions(Y_test, Y_pred, sensor_idx=0, num_samples=100):
    """
    Plots actual vs. predicted values for a selected sensor.

    Args:
        Y_test (np.array): True labels.
        Y_pred (np.array): Model predictions.
        sensor_idx (int): Index of sensor to plot.
        num_samples (int): Number of time steps to visualize.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test[:num_samples, sensor_idx], label="Actual", color="blue")
    plt.plot(Y_pred[:num_samples, sensor_idx], label="Predicted", color="red", linestyle="dashed")
    plt.xlabel("Time Steps")
    plt.ylabel("Traffic Speed")
    plt.title(f"Linear Regression Predictions for Sensor {sensor_idx}")
    plt.legend()
    plt.savefig('convertors/images/metr-la_lin_reg.pdf')
    plt.show()

if __name__ == "__main__":
    # Filepath to processed traffic dataset
    filepath = "convertors/Datasets/METR-LA.h5"  

    # Load and normalize
    df_scaled, scaler = load_and_normalize_data(filepath)

    # Create input-output pairs
    X, Y = create_sliding_window(df_scaled, lookback=12, predict_forward=1)

    # Train model
    model, X_test, Y_test, Y_pred = train_linear_regression(X, Y)

    # Evaluate model
    metrics = evaluate_model(Y_test, Y_pred)
    print(f"🔹 MSE: {metrics['MSE']:.4f}")
    print(f"🔹 R² Score: {metrics['R2']:.4f}")

    # Plot results for a specific sensor
    plot_predictions(Y_test, Y_pred, sensor_idx=0)