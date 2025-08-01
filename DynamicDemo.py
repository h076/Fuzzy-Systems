"""
    - Load data with labels
    - Initialise membership functions
    - visualise initial membership functions
    - generate rule base using data
    - train model through rule base generation
    - visualise final membership functions
    - evaluate model using test set
    - provide interpretable response to all test set predictions
    - save model state
"""

import numpy as np
import pandas as pd
from typing import Any

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import membershipFunctions as mf
import ANFIS as anfis

#import RuleGenDynamic as rgd
import FinalRuleGen as frg
import RulesToNatural as nat
import EvaluateModel as eval
import SaveAndLoad as snl

def main():
    dfSamples = pd.read_csv("Data/Samples.csv")
    npSamples = dfSamples.to_numpy()

    ranges = np.array((npSamples[0], npSamples[1])).T
    x_ranges = ranges[:-2]
    y_range: tuple[Any, Any] = (ranges[-2][0], ranges[-2][1])

    x_data = npSamples[2:, :-2].astype(float)
    y_data = npSamples[2:, -2].astype(float)
    y_labels = npSamples[2:, -1]

    feature_names = dfSamples.columns.values.tolist()[:-2]

    # visualise initial membership functions
    input_mfs = nn.ModuleList()
    for (min_val, max_val), name in zip(x_ranges, feature_names):
        params = mf.generateBellParams(5, (min_val, max_val))
        imf = mf.TrainableBellMF(params, name)
        imf.visualise((min_val, max_val))
        input_mfs.append(imf)

    params = mf.generateOutputMFParams(y_range)
    output_mf = mf.TrainableTrapezoidMF(params)
    output_mf.visualise(y_range)

    model = anfis.MamdaniANFIS(input_mfs, output_mf, y_range, {})

    X_train, X_test, y_train_data, y_test_data, y_train_labels, y_test_labels = train_test_split(
        x_data, y_data, y_labels, test_size=0.2, random_state=42
    )

    # generate rule base and train model
    model = frg.ruleGeneration(X_train, x_ranges, y_train_data, y_range, model, {})

    # visualise final membership functions
    for (min_val, max_val), input_mf in zip(x_ranges, input_mfs):
        input_mf.visualise((min_val, max_val))

    output_mf.visualise(y_range)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test_data, dtype=torch.float32)

    # evaluate MSE
    eval.evaluateMSE(model, X_test, y_test)

    # provide interpretable response to predictions
    natural_parser = nat.RulesToNatural(y_range, feature_names)
    predictions_firing_rules = model.getPredictions(X_test)

    for (prediction, rules), y in zip(predictions_firing_rules, y_data):
        print(y)
        natural_parser.explainPrediction(rules, prediction)

    # Save model parameters
    snl.saveModel(model, "gen_demo_model_params.pth")

if __name__ == "__main__":
    main()