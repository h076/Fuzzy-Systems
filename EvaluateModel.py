import string

import numpy as np
import torch
import torch.nn as nn

def evaluateMSE(model: nn.Module, x_data: torch.Tensor, y_data: torch.Tensor):
    # Test predictions
    predictions = model(x_data).detach()
    # Calculate MSE using torch operations
    mse = torch.mean((y_data - predictions) ** 2).item()

    print("model MSE : {0:.4f}".format(mse))

def evaluateClassification(model: nn.Module, x_data: torch.Tensor, y_labels: [string], y_range: (float, float)):
    # Test predictions
    predictions = model(x_data).detach()

    correct_predictions = 0
    total_predictions = 0

    for p, signal in zip(predictions, y_labels):
        if signal == getLabel(p, y_range):
            correct_predictions += 1
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print("model classification accuracy : {0:.2f}".format(accuracy))

def getLabel(prediction: float, y_range: (float, float)):
    min_range = y_range[0]
    max_range = y_range[1]

    label = ""
    if prediction > 0.5 * max_range:
        label = "strong buy"
    elif prediction >= 0.25 * max_range:
        label = "buy"
    elif prediction >= 0.25 * min_range:
        label = "hold"
    elif prediction < 0.25 * min_range:
        label = "sell"
    elif prediction < 0.5 * min_range:
        label = "strong sell"

    return label
