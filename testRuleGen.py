import pandas as pd
import numpy as np

import ANFIS
import membershipFunctions as mf
from ANFIS import MamdaniANFIS
import torch
import NewRuleGen as rg

from sklearn.model_selection import train_test_split

dfSamples = pd.read_csv('Data/Samples.csv')
npSamples = dfSamples.to_numpy()

ranges = np.array((npSamples[0], npSamples[1])).T
xRanges = ranges[:-1]
yRange = ranges[-1]

xData = npSamples[2:, :-2]
yData = npSamples[2:, -2]
#labels = npSamples[2:, -1]

inputMFs = []
for minVal, maxVal in xRanges:
    params = mf.generateBellParams(5, (minVal, maxVal))
    imf = mf.TrainableBellMF(params)
    # imf.visualise((minVal, maxVal))
    inputMFs.append(imf)

params = mf.generateTrapezoidParams(5, yRange)
outputMF = mf.TrainableTrapezoidMF(params)
# outputMF.visualise(yRange)

model = MamdaniANFIS(inputMFs, outputMF, yRange, {})

X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=0.2, random_state=42)

# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)

model = rg.ruleGeneration(X_train, xRanges, y_train, yRange, model, {})

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_loss = ANFIS.getModelLoss(model, X_test, y_test)
model.explainPrediction(X_test, y_test)
