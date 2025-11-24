# Advanced-Time-Series-Forecasting-with-Attention-based-Neural-Networks
Advanced Time Series Forecasting with Attention-based Neural Networks

This project implements multivariate time series forecasting using a baseline LSTM model and an attention-based encoder–decoder neural network. The script is completely Colab-ready and runs in a single code cell.

Features

Synthetic multivariate time series dataset generation

MinMax scaling for all features

Sliding-window sequence creation

Baseline LSTM model

Bahdanau Attention model (custom implementation)

Training with EarlyStopping and ReduceLROnPlateau

Prediction plots for scaled and original values

Attention weight visualization

Model and prediction file saving

Dataset Description

A synthetic dataset with 1400 samples and 4 columns:

feature1: trend + sinusoidal pattern + noise

feature2: sinusoidal + noise

feature3: random walk + slow trend

target: weighted combination of all features + noise

The target variable is predicted one step ahead using the past 36 timesteps.

Model Architectures
Baseline LSTM

LSTM layer with 64 units

Dense(32) + Dense(1)

Optimizer: Adam

Loss: MSE

Attention Model

Encoder LSTM with return sequences

Custom Bahdanau Attention layer

Context vector concatenated with hidden state

Dense layers for final output

The attention model provides both predictions and attention weights for interpretability.

Training Details

Both models are trained using:

EarlyStopping (patience=6)

ReduceLROnPlateau (patience=3)

80% training split, 20% test split

Batch size: 32

Epochs: 40–60

The script outputs:

MAE and RMSE for both models

Plots comparing actual vs predicted values

Attention heatmap for a sample test sequence

Results

The attention model typically performs better than the baseline LSTM, producing lower MAE and RMSE. The script prints metrics after testing and shows plots to visually compare performance.

Output Files

The script automatically saves the following:

models/baseline_lstm.h5

models/attention_model.h5

predictions_scaled.csv

These files can be used for reports, further training, or deployment.

How to Run

Open Google Colab

Copy the entire Python script

Paste into a new code cell

Run the cell

All models, metrics, and plots are generated automatically.

Possible Extensions

Multi-step forecasting

Transformer-based architectures

Hyperparameter tuning

Additional explainability (e.g., SHAP)
