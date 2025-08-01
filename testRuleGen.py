import pandas as pd
import numpy as np
import torch.nn as nn

import ANFIS as anfis
import ANFISTopNRule as anfisT
import membershipFunctions as mf
from ANFIS import MamdaniANFIS
import torch
import NewRuleGen as rg

from sklearn.model_selection import train_test_split

dfSamples = pd.read_csv('Data/Samples.csv')
npSamples = dfSamples.to_numpy()

ranges = np.array((npSamples[0], npSamples[1])).T
xRanges = ranges[:-2]
yRange = ranges[-2]

xData = npSamples[2:, :-2].astype(float)
yData = npSamples[2:, -2].astype(float)
labels = npSamples[2:, -1]

inputMFs = nn.ModuleList()
for minVal, maxVal in xRanges:
    params = mf.generateBellParams(5, (minVal, maxVal))
    imf = mf.TrainableBellMF(params)
    #imf.visualise((minVal, maxVal))
    inputMFs.append(imf)

params = mf.generateTrapezoidParams(5, yRange)
outputMF = mf.TrainableTrapezoidMF(params)
# outputMF.visualise(yRange)

model = anfisT.MamdaniANFIS(inputMFs, outputMF, yRange, {})

X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=42)

#y_test_labels = y_test[:-1]
#y_test = y_test[:0].astype(float)

#y_train = y_train[:0].astype(float)

#X_train = X_train.astype(float)
#X_test = X_test.astype(float)

#print(X_train)
#print(y_train)

# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)

model = rg.ruleGeneration(X_train, xRanges, y_train, yRange, model, {})

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_loss = anfis.getModelLoss(model, X_test, y_test)
model.explainPrediction(X_test, y_test)
