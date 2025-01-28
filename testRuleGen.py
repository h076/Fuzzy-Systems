import pandas as pd
import numpy as np
import membershipFunctions as mf
from ANFIS import MamdaniANFIS
import torch
import NewRuleGen as rg

dfSamples = pd.read_csv('Data/Samples.csv')
npSamples = dfSamples.to_numpy()

ranges = np.array((npSamples[0], npSamples[1])).T
xRanges = ranges[:-1]
yRange = ranges[-1]

xData = npSamples[2:, :-1]
yData = npSamples[2:, -1]

inputMFs = []
for minVal, maxVal in xRanges:
    params = mf.generateBellParams(5, (minVal, maxVal))
    imf = mf.TrainableBellMF(params)
    #imf.visualise((minVal, maxVal))
    inputMFs.append(imf)

params = mf.generateTrapezoidParams(5, yRange)
outputMF = mf.TrainableTrapezoidMF(params)
#outputMF.visualise(yRange)

model = MamdaniANFIS(inputMFs, outputMF, yRange, {})
X = torch.tensor(xData, dtype=torch.float32)
y = torch.tensor(yData, dtype=torch.float32)

model = rg.ruleGeneration(xData, xRanges, yData, yRange, model, {})

