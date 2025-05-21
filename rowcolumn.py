import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import os

# Enable mixed precision for speed
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Ensure TensorFlow uses GPU memory efficiently
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Z-score normalization
def z_score_normalization(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data.fillna(0))
    return normalized_data, scaler

# Rescale to original range
def rescale_to_original(data, scaler):
    return (data * scaler.scale_) + scaler.mean_

# Add cyclical features
def add_cyclical_features(data, Timestamp_column):
    data['month'] = pd.to_datetime(data[Timestamp_column]).dt.month
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    return data

# Build Temporal Convolutional Network
def build_tcn(input_shape):
    inputs = Input(shape=input_shape)
    tcn = Conv1D(64, 3, dilation_rate=1, activation='relu', padding='causal')(inputs)
    tcn = Conv1D(128, 3, dilation_rate=2, activation='relu', padding='causal')(tcn)
    tcn = Conv1D(input_shape[-1], 1, activation='linear', padding='causal')(tcn)  # Match output features to input
    return Model(inputs, tcn, name="TCN")

# Build Denoising Autoencoder
def build_denoising_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(32, activation='relu', return_sequences=False)(inputs)
    bottleneck = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(bottleneck)
    outputs = TimeDistributed(Dense(input_shape[1], activation="linear"))(decoded)
    autoencoder = Model(inputs, outputs, name="DenoisingAutoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Resample Timestamp column
def resample_data(data, Timestamp_column, frequency='15T'):
    """
    Resamples the data to the specified frequency based on the Timestamp column.

    Args:
    data (pd.DataFrame): Input DataFrame.
    Timestamp_column (str): Name of the timestamp column.
    frequency (str): Resampling frequency (e.g., '15T' for 15 minutes).

    Returns:
    pd.DataFrame: Resampled DataFrame with missing timestamps filled as NaN.
    """
    data[Timestamp_column] = pd.to_datetime(data[Timestamp_column])
    data.set_index(Timestamp_column, inplace=True)
    resampled_data = data.resample(frequency).mean()  # Resample and maintain numeric data
    resampled_data.reset_index(inplace=True)
    return resampled_data

def identify_large_gaps(data, Timestamp_column, gap_threshold='30D'):
    """
    Identifies large gaps in a dataset based on a time threshold.

    Args:
    data (pd.DataFrame): Input dataset with a timestamp column.
    Timestamp_column (str): Name of the timestamp column in the dataset.
    gap_threshold (str): Threshold for detecting large gaps (e.g., '30D' for 30 days).

    Returns:
    pd.DataFrame: The input DataFrame with an additional column 'large_gap'.
                  The 'large_gap' column is True for rows where a large gap is detected.
    """
    try:
        # Ensure the timestamp column is in datetime format
        data[Timestamp_column] = pd.to_datetime(data[Timestamp_column])

        # Sort data by the timestamp column to ensure proper gap detection
        data = data.sort_values(by=Timestamp_column)

        # Compute time differences between consecutive rows
        time_diffs = data[Timestamp_column].diff()

        # Detect large gaps based on the threshold
        data['large_gap'] = time_diffs > pd.Timedelta(gap_threshold)

        # Handle the first row (no previous row for comparison)
        data['large_gap'].iloc[0] = False  # The first row cannot have a large gap

        return data

    except Exception as e:
        print(f"Error identifying large gaps: {e}")
        return data


def preprocess_data(data):
    """
    Preprocesses a dataset for missing value imputation.
    - Identifies numeric and non-numeric columns.
    - Normalizes numeric data using Z-score normalization.
    - Creates a mask for missing values.
    - Reshapes data for model input.

    Args:
    data (pd.DataFrame): Input dataset.

    Returns:
    tuple:
        - reshaped_data (np.ndarray): Normalized and reshaped data.
        - reshaped_mask (np.ndarray): Mask for missing values (1 for missing, 0 otherwise).
        - scaler (StandardScaler): Fitted scaler for inverse normalization.
        - non_numeric_columns (list): List of non-numeric column names in the input data.
    """
    try:
        # Step 1: Identify non-numeric columns
        non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

        # Step 2: Extract numeric data
        numeric_data = data.drop(columns=non_numeric_columns, errors='ignore')

        # Step 3: Normalize numeric data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(numeric_data.fillna(0))

        # Step 4: Create a mask for missing values
        missing_mask = numeric_data.isnull().astype(int).values

        # Step 5: Reshape data for model input
        num_rows, num_cols = normalized_data.shape
        reshaped_data = normalized_data.reshape((num_rows, 1, num_cols))
        reshaped_mask = missing_mask.reshape((num_rows, 1, num_cols))

        return reshaped_data, reshaped_mask, scaler, non_numeric_columns

    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return None, None, None, None


# Process a single chunk
def process_chunk(chunk, autoencoder, tcn_model, Timestamp_column, gap_threshold='30D'):
    """
    Process a single chunk:
    - Identify large gaps.
    - Fill small gaps using TCN and Autoencoder while leaving rows with large gaps unchanged.

    Args:
    chunk (pd.DataFrame): Data chunk to process.
    autoencoder (Model): Trained autoencoder.
    tcn_model (Model): Trained TCN.
    Timestamp_column (str): Name of the timestamp column.
    gap_threshold (str): Threshold for large gaps.

    Returns:
    pd.DataFrame: Imputed DataFrame with only small gaps filled.
    """
    try:
        # Resample data to 15-minute intervals
        chunk = resample_data(chunk, Timestamp_column, frequency='15T')

        # Identify large gaps
        chunk = identify_large_gaps(chunk, Timestamp_column, gap_threshold)

        # Separate rows with large gaps
        large_gap_data = chunk[chunk['large_gap']]
        small_gap_data = chunk[~chunk['large_gap']].copy()

        # Preprocess data for small gaps
        reshaped_data, reshaped_mask, scaler, non_numeric_columns = preprocess_data(small_gap_data)

        # Pass reshaped data through TCN
        tcn_features = tcn_model.predict(reshaped_data, batch_size=1024, verbose=0)

        # Ensure TCN output matches Autoencoder input shape
        if tcn_features.shape[-1] != reshaped_data.shape[-1]:
            tcn_features = tcn_features[:, :, :reshaped_data.shape[-1]]

        # Reconstruct missing values with autoencoder
        reconstructed_data = autoencoder.predict(tcn_features, batch_size=1024, verbose=0)

        # Impute missing values
        imputed_data = reshaped_data.copy()
        imputed_data[reshaped_mask == 1] = reconstructed_data[reshaped_mask == 1]

        # Rescale to original range
        imputed_data_rescaled = rescale_to_original(imputed_data.reshape(-1, imputed_data.shape[-1]), scaler)

        # Combine imputed small gaps and rows with large gaps
        numeric_columns = small_gap_data.drop(columns=non_numeric_columns).columns
        imputed_df = pd.DataFrame(imputed_data_rescaled, columns=numeric_columns)
        imputed_df[non_numeric_columns] = small_gap_data[non_numeric_columns].reset_index(drop=True)

        # Reintegrate large-gap rows into the final DataFrame
        final_df = pd.concat([imputed_df, large_gap_data], ignore_index=True).sort_values(by=Timestamp_column)

        return final_df

    except Exception as e:
        print(f"Error processing chunk: {e}")
        print(f"Chunk shape: {chunk.shape}")
        return chunk  # Return the original chunk in case of failure


# Main function
def fill_missing_values_in_chunks(file_path, Timestamp_column, output_file, chunk_size=100000):
    if os.path.exists(output_file):
        os.remove(output_file)

    # Load one chunk to initialize models
    first_chunk = pd.read_csv(file_path, chunksize=chunk_size).__next__()
    input_shape = (1, first_chunk.select_dtypes(include=[np.number]).shape[1])
    tcn_model = build_tcn(input_shape)
    autoencoder = build_denoising_autoencoder(tcn_model.output_shape[1:])

    for chunk_id, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        print(f"Processing chunk {chunk_id}...")
        try:
            imputed_chunk = process_chunk(chunk, autoencoder, tcn_model, Timestamp_column)

            # Save chunk to output file
            if not imputed_chunk.empty:
                imputed_chunk.to_csv(output_file, mode='a', header=(chunk_id == 0), index=False)
            print(f"Chunk {chunk_id} processed and saved.")
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            continue

    print(f"Imputed dataset saved to '{output_file}'.")

# Example usage
if __name__ == "__main__":
    file_path = "F:\\AQI\\City-Delhi\\Merged Data for each station - delhi\\station_Okhla phase - 2.csv"
    output_file = "F:\\AQI\\City-Delhi\\okhla\\imputed_station_Okhla phase - 2.csv"
    fill_missing_values_in_chunks(file_path, "Timestamp", output_file, chunk_size=500000)