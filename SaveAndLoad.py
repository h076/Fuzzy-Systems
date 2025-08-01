import string

import torch
import torch.nn as nn

import membershipFunctions as mf
import ANFIS as anfis

def saveModel(model: nn.Module, path: string):
    torch.save({
        'model_state_dict': model.state_dict(),
        'rule_base': model.rule_base,
        'output_range': model.output_range
    }, path)

    print("saved model state to : {0}".format(path))

def loadModel(x_ranges: [(float, float)], y_range: (float, float), path: string):
    # Load checkpoint
    checkpoint = torch.load(path)

    # initialise membership functions
    inputMFs = nn.ModuleList()
    for minVal, maxVal in x_ranges:
        params = mf.generateBellParams(5, (minVal, maxVal))
        imf = mf.TrainableBellMF(params)
        inputMFs.append(imf)

    params = mf.generateTrapezoidParams(5, y_range)
    outputMF = mf.TrainableTrapezoidMF(params)

    # initialise new model with plain membership functions
    new_model = anfis.MamdaniANFIS(
        input_mfs=inputMFs,
        output_mf=outputMF,
        output_range=checkpoint['output_range'],
        rule_base=checkpoint['rule_base']
    )

    # Restore model parameters
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_model.eval()

    return new_model

def loadToModel(model: nn.Module, path: string):
    # Load checkpoint
    checkpoint = torch.load(path)

    # Restore model parameters
    model.load_state_dict(checkpoint['model_state_dict'])
