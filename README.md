# Alloy Corrosion Prediction with Physics Guided Neural Networks

Welcome to the **Alloy Corrosion Prediction** repository! This project leverages neural networks to predict material properties, featuring both a Physics-Constrained Neural Network (PCNN) and a Traditional Neural Network (NN) for comprehensive analysis.

This project requires **Python 3.8 or higher**. 

## Features

- Data Processing: Cleans and preprocesses data from `New_Modified_Copy.xlsx`.
- Custom Datasets: Utilizes PyTorch's `Dataset` and `DataLoader` for efficient data handling.
- Models:
  - Physics-Constrained NN (PCNN): Integrates physical laws to enhance prediction accuracy.
  - Traditional NN: Serves as a baseline for performance comparison.
- Training & Evaluation: Includes scripts for training models, calculating Mean Squared Error (MSE), Percentage Errors, and generating comparison plots.
- Visualization: Generates plots to compare predicted values against actual targets.
- Model Saving: Saves trained models for future use.

## Install Dependencies:

pip install -r requirements.txt

## Prepare Data:

Ensure the dataset file corrosion_data.xlsx is placed in the root directory of the project.

## Train Models:

Physics-Constrained NN:

Alloy_PINN.py

Traditional NN:

Alloy_NN.py

