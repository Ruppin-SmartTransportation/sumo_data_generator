import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------- Data Preprocessing ---------------------- #
def create_sliding_window(data, seq_length=10):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        Y.append(data[i + seq_length])
    return np.array(X), np.array(Y)

# ---------------------- Load & Prepare Data ---------------------- #
def load_and_prepare_data(file_path, seq_length=10):

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
    data_scaled = scaler.fit_transform(df)
    
    X, Y = create_sliding_window(data_scaled, seq_length)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    return X_train, X_test, Y_train, Y_test, scaler

# ---------------------- Build LSTM Model ---------------------- #
def build_lstm_model(input_shape, output_dim):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------- Train Model ---------------------- #
def train_model(model, X_train, Y_train, epochs=50, batch_size=32):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return model

# ---------------------- Save Model ---------------------- #
def save_model(model, file_path="convertors/models/lstm_metr-la_model.h5"):
    model.save(file_path)
    print(f"Model saved to {file_path}")

# ---------------------- Load Model ---------------------- #
def load_trained_model(file_path="convertors/models/lstm_metr-la_model.h5"):
    return load_model(file_path)

# ---------------------- Evaluate Model ---------------------- #
def evaluate_model(model, X_test, Y_test, scaler):
    Y_pred = model.predict(X_test)
    Y_test_inv = scaler.inverse_transform(Y_test)
    Y_pred_inv = scaler.inverse_transform(Y_pred)
    
    mse = mean_squared_error(Y_test_inv, Y_pred_inv)
    r2 = r2_score(Y_test_inv, Y_pred_inv)
    
    print(f"🔹 MSE: {mse:.4f}")
    print(f"🔹 R² Score: {r2:.4f}")
    return Y_test_inv, Y_pred_inv

# ---------------------- Plot Predictions ---------------------- #
def plot_predictions(Y_test, Y_pred, sensor_idx=188, num_samples=100):
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
    plt.legend()
    plt.title("Traffic Speed Prediction - LSTM")
    plt.savefig(f'convertors/images/metr-la_lstm_{sensor_idx}.pdf')
    plt.show()

# ---------------------- Run the Pipeline ---------------------- #
if __name__ == "__main__":
    file_path = "convertors/Datasets/METR-LA.h5"  
    seq_length = 10
    X_train, X_test, Y_train, Y_test, scaler = load_and_prepare_data(file_path, seq_length)
    
    #model = build_lstm_model(input_shape=(seq_length, X_train.shape[2]), output_dim=Y_train.shape[1])
    model = load_trained_model()
    # model = train_model(model, X_train, Y_train)
    # save_model(model)

    Y_test_inv, Y_pred_inv = evaluate_model(model, X_test, Y_test, scaler)
    plot_predictions(Y_test_inv, Y_pred_inv)
