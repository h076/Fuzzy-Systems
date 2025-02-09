#!/usr/bin/env python3

from functools import reduce
from operator import mul
import numpy as np
import torch.nn as nn
import torch
from torch import optim
from typing import Dict

import NewRuleGen as rg

"""
class inputLayer(nn.Module):

    def __init__(self, fn, d, dr):
        # List of fetaure identifiers
        self.featureNames = fn

        # Pandas data frame containing data,
        # including feature identifiers
        self.data = d

        # Dictionary of (min, max) tuples
        # Key = feature identifier
        self.dataRanges = dr

        # Number of data samples
        self.dataCount = self.data.size()

    def getDataCount(self):
        return self.dataCount

    def getFeatureData(self, n):
        if n not in self.featureNames:
            print("inputLayer.getFeatureData: feature not present")

        return self.data[n]


class fuzzificationLayer(nn.Module):

    # input collection of membership function for each input
    # number of inputs
    def __init__(self, bellCollections, numInputs):
        super().__init__()

        self.numInputs = numInputs
        self.inputMFs = nn.ModuleList()  # List to store membership function nodes for each input
        for bc in bellCollections:
            self.inputMFs.append(nn.ModuleList(bc))

    # input : tensor of input values
    def foward(self, inputs) -> torch.Tensor.list().list():
        # fuzzify all inputs to get their membership value
        fuzzyInputs = []
        for i in range(self.numInputs):
            inputValue = inputs[i]
            fuzzyInput = [mf.foward(inputValue) for mf in self.inputMfs[i]]
            fuzzyInputs.append(fuzzyInput)

        # return array of dictionaries containing MFNumber, Fuzzified value pairs
        return fuzzyInputs


class fuzzLayer(nn.Module):
    def __init__(self, inputMfs, numInputs):
        super().__init__()
        self.inputMfs = inputMfs
        self.numInputs = numInputs

    def foward(self, inputs):
        fuzzyOutput = []
        for i in range(self.numInputs):
            fuzzyOutput.append(self.inputMfs[i].forward(inputs[i]))
        return fuzzyOutput


class inferenceLayer(nn.Module):

    # input ruleBase : is array of dictionarys of input index, MFNumber pairs
    # input MFBase : [MFgroups]
    def __init__(self, ruleBase, MFBase):
        super().__init__()
        self.nodes = []
        self.ruleBase = ruleBase
        self.MFBase = MFBase
        self.createNodesFromRules()

    # Create new nodes for each rule
    # ruleDict is dictionary of input index, MFNumber pairs
    def createNodesFromRules(self):
        for rule in self.ruleBase:
            antecedents, consequent = rule
            # create node with antecedents only, not consequent
            self.nodes.append(self.inferenceNode(antecedents, consequent))

    def foward(self, fuzzifiedInputs: torch.Tensor.list().list()) -> (torch.Tensor, torch.Tensor).list():
        outputs = []
        for n in self.nodes:
            outputs.append(n.foward(fuzzifiedInputs, self.MFBase))

        return outputs

    # contains
    class inferenceNode:

        # input antecedant : [(MFGroupIdx, MFIdx)]
        # input consequent : (MFGroupIdx, MFIdx)
        def __init__(self, antecedents, consequent):
            self.antecedents = antecedents
            self.consequent = consequent

        def foward(self, fuzzifiedInputs: torch.Tensor.list().list(),
                   mfBase: MembershipFunctionGroup.list()) -> (torch.Tensor, torch.Tensor):

            antecedantValues = []
            for inputIdx, mfIdx in self.antecedents:
                antecedantValues.append(fuzzifiedInputs[inputIdx][mfIdx])

            mfGroupIdx, mfIdx = self.consequent
            mfGroup = mfBase[mfGroupIdx]
            outputMf = mfGroup.getMF(mfIdx);
            return (torch.min(antecedantValues), outputMf)


class infLayer(nn.Module):
    def __init__(self, ruleBase, outputMfs):
        super().__init__()
        self.nodes = []
        self.ruleBase = ruleBase
        self.outputMfs = outputMfs

    def foward(self, fuzzifiedInputs) -> (torch.Tensor.list(), TrapezoidMF):
        output = []
        for rule in self.ruleBase:
            strengths = [fuzzifiedInputs[i][m] for i, m in rule['antecedent']]
            output.append((torch.min(strengths), self.outputMfs[rule['consequent']]))

        return output


class ImplicationLayer(nn.Module):

    def __init__(self, numRules: int):
        super().__init__()

        self.nodes = nn.ModuleList()


    def foward(self, firingStrengths: torch.Tensor.list(), mfs: torch.Tensor.list()) -> torch.Tensor.list():
        if self.nodes.empty():
            for mf in mfs:
                self.nodes.append(self.implicationNode(mf))

        outputMfs = []
        for i in range(len(firingStrengths)):
            n = self.nodes[i]
            outputMfs.append(n.foward(firingStrengths[i]))

        return outputMfs

    class implicationNode:

        def __init__(self, outputMf: torch.Tensor = None):
            self.outputMf = outputMf

        def foward(self, firingStrength: float, mf: torch.Tensor = None, method: str = "prod") -> torch.Tensor:
            if self.outputMf is None:
                self.outputMf = mf

            if method == "min":  # Clipping method
                return np.minimum(firingStrength, self.outputMf)
            elif method == "prod":  # Scaling method
                return firingStrength * self.outputMf
            else:
                raise ValueError("Invalid implication method. Use 'min' or 'prod'.")

class impLayer(nn.Module):
    def __init__(self, outputMf):
        self.outputMf = outputMf

    def forward(self, firingStrength, mfIdx, method):
        if method == "prod":  # Scaling method
            self.outputMf.scale_params(mfIdx, firingStrength)
            return self.outputMf.get_membership_function(mfIdx)
        else:
            raise ValueError("Invalid implication method. Use 'prod'.")


class ApplicationLayer(nn.Module):

    def __init__(self):
        super().__init__()

    # input outputs : list of outputs from Implication layer
    def forward(self, outputs):
        # take the element wise maximum
        return torch.max(torch.stack(outputs), dim=0)[0]

class DeffuzificationLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, aggregated_mf, output_universe):

        # Compute the weighted sum (numerator)
        numerator = torch.sum(aggregated_mf * output_universe, dim=1)

        # Compute the sum of membership values (denominator)
        denominator = torch.sum(aggregated_mf, dim=1)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-6
        crisp_output = numerator / (denominator + epsilon)

        return crisp_output
"""

def getModelLoss(model, x_data: torch.Tensor, y_data: torch.Tensor):
    # Test predictions
    predictions = model(x_data).detach()
    # Calculate MSE using torch operations
    return torch.mean((y_data - predictions) ** 2).item()

class MamdaniANFIS(nn.Module):
    def __init__(self, input_mfs, output_mf, output_range, rule_base):
        super().__init__()
        self.input_membership_functions = input_mfs
        self.output_membership_function = output_mf
        self.rule_base = rule_base
        self.output_range = output_range
        self.epsilon = 1e-5

        self.rule_firing = []
        self.rule_firing_indices = []

    def set_rule_base(self, rule_base):
        self.rule_base = rule_base

    def explainPrediction(self, x_data: torch.Tensor, y_data: torch.Tensor):
        predictions = self.forward(x_data)
        for idx, p in enumerate(predictions):
            print("predicted : {0}, True : {1}".format(p, y_data[idx]))
            print("Rule firing ...")
            significant_rules = {}
            minimum_firing = 10000
            print(len(self.rule_firing[idx]))
            for i, r in enumerate(self.rule_firing[idx]):
                if len(significant_rules) == 8:
                    if minimum_firing < r:
                        significant_rules.pop(minimum_firing)
                        significant_rules[r] = i
                        minimum_firing = min(significant_rules.keys())
                else:
                    significant_rules[r] = i
                    minimum_firing = min(minimum_firing, r)

            print("Significant rules : ")
            #for firing, index in significant_rules.items():
                #print("Rule {0} fired at : {1}".format(index, firing))
                #rg.printRule(self.rule_base[self.rule_firing_indices[index]])
            rules = []
            for firing, index in significant_rules.items():
                rules.append(self.rule_base[self.rule_firing_indices[index]])
            rg.explainOutcome(rules, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Fuzzification layer
        fuzzy_inputs = []
        for i, mf in enumerate(self.input_membership_functions):
            fuzzy_values = mf(x[:, i])
            fuzzy_values = torch.clamp(fuzzy_values, min=self.epsilon, max=1.0)
            fuzzy_inputs.append(fuzzy_values)

        # Inference layer
        # Firing strengths generated using the product method
        rule_firing_strengths = []
        firing_indices = []
        for rule_key, rule in self.rule_base.items():
            mf_indices = rule['antecedent']
            firing_strength = torch.ones(batch_size, device=x.device)
            for input_idx, mf_idx in enumerate(mf_indices):
                if mf_idx != -1:
                    current_membership = fuzzy_inputs[input_idx][mf_idx]
                    firing_strength *= current_membership  # Element-wise multiplication

            rule_firing_strengths.append(firing_strength)
            firing_indices.append(mf_indices)

        # Stack all rule firing strengths into tensor of shape (batch_size, num_rules)
        rule_firing_strengths = torch.stack(rule_firing_strengths, dim=1)
        self.rule_firing = rule_firing_strengths
        self.rule_firing_indices = firing_indices

        # Implication layer
        # Assume a discrete universe of discourse for output
        num_points = 100
        output_universe = torch.linspace(self.output_range[0], self.output_range[1], num_points, device=x.device)
        implications = torch.zeros(batch_size, len(self.rule_base), num_points, device=x.device)

        for rule_idx, (rule_key, rule) in enumerate(self.rule_base.items()):
            # get output membership function index
            mf_out_idx = rule['consequent']

            # shape : [num_points]
            output_memberships = self.output_membership_function(output_universe)[mf_out_idx]

            # Apply product implication, multiply firing strengths by memberships
            # expand firing strengths to match output membership shape
            firing_strength = rule_firing_strengths[:, rule_idx].unsqueeze(1)
            implication = firing_strength * output_memberships.unsqueeze(0)

            implications[:, rule_idx] = implication

        # Aggregation layer
        # use sum operator to sum all rule implications along all rule dimensions
        aggregated_output = torch.sum(implications, dim=1) # Shape: [batch_size, num_points]

        # Defuzzification layer using center of mass
        # Avoid division by zero by adding small epsilon
        numerator = torch.sum(output_universe * aggregated_output, dim=1)  # Shape: [batch_size]
        denominator = torch.sum(aggregated_output, dim=1) + self.epsilon  # Shape: [batch_size]
        crisp_output = numerator / denominator  # Shape: [batch_size]

        return crisp_output

    def train_model(self, train_data: torch.Tensor,
                    train_labels: torch.Tensor,
                    num_epochs: int = 50,
                    learning_rate: float = 0.002,
                    batch_size: int = 32,
                    rule_base: Dict = None,
                    progress: bool = False,
                    progress_interval: int = 10):

        if rule_base:
            self.set_rule_base(rule_base)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            indices = torch.randperm(len(train_data))
            train_data = train_data[indices]
            train_labels = train_labels[indices]

            total_loss = 0
            num_batches = len(train_data) // batch_size

            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                batch_data = train_data[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]

                optimizer.zero_grad()
                outputs = self(batch_data)

                loss = criterion(outputs, batch_labels)

                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {batch}")
                    return

                loss.backward()

                # Add both gradient clipping methods
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0) ###

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)

            if epoch % progress_interval == 0 and progress:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")