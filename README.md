# 📈 Time-Series Missing Value Imputation with TCN + LSTM Autoencoder

This project provides a scalable solution to **fill missing values in large time-series datasets** using a combination of **Temporal Convolutional Networks (TCNs)** and **LSTM-based Denoising Autoencoders**.

It is tailored for environmental or sensor data, such as air quality datasets, with large gaps, irregular timestamps, and complex seasonal patterns.


## 🧠 Core Features

- ⌛ **Timestamp Resampling**: Standardizes irregular timestamps into uniform intervals (e.g., 15 minutes).
- 🚨 **Large Gap Detection**: Flags gaps larger than a configurable threshold (e.g., 30 days).
- 🧼 **Small Gap Imputation**:
  - Uses a Temporal Convolutional Network (TCN) to extract temporal features.
  - Applies a Denoising LSTM Autoencoder to reconstruct and fill missing values.
- 🔄 **Chunked Processing**: Processes large files efficiently in chunks (default: 500,000 rows).
- 📦 **Z-score Normalization** and **Inverse Scaling** to maintain data integrity.
- 🔁 **Cyclical Encoding** for month information to capture seasonality.


## 🏗️ Architecture Overview

1. **Input CSV Chunk →**
2. **Resampling + Gap Detection →**
3. **TCN Feature Extraction →**
4. **LSTM Autoencoder Reconstruction →**
5. **Missing Value Imputation →**
6. **Output Saved to Final CSV**

