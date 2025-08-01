import torch.nn as nn
import torch
from torch import optim
from typing import Dict

import NewRuleGen as rg
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

        self.top_n = 8

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

        # Stack all rule firing strengths into a tensor (batch_size, num_rules)
        rule_firing_strengths = torch.stack(rule_firing_strengths, dim=1)

        # Select top-N firing rules per batch
        top_values, top_indices = torch.topk(rule_firing_strengths, self.top_n, dim=1)  # Shape: (batch_size, top_n)

        # Save rule firings for explanation
        self.rule_firing = top_values
        self.rule_firing_indices = [[firing_indices[idx.item()] for idx in batch_indices] for batch_indices in top_indices]

        # Implication layer: Compute output based on top-N rules
        num_points = 100
        output_universe = torch.linspace(self.output_range[0], self.output_range[1], num_points, device=x.device)
        implications = torch.zeros(batch_size, self.top_n, num_points, device=x.device)

        for i in range(self.top_n):
            rule_idx = top_indices[:, i]
            firing_strength = top_values[:, i].unsqueeze(1)

            for batch_idx in range(batch_size):
                rule_key = firing_indices[rule_idx[batch_idx].item()]
                mf_out_idx = self.rule_base[rule_key]['consequent']
                output_memberships = self.output_membership_function(output_universe)[mf_out_idx]

                implications[batch_idx, i] = firing_strength[batch_idx] * output_memberships

        # Aggregation: Sum up implications of selected top-N rules
        aggregated_output = torch.sum(implications, dim=1)  # Shape: [batch_size, num_points]

        # Defuzzification: Compute final crisp output using center of mass
        numerator = torch.sum(output_universe * aggregated_output, dim=1)
        denominator = torch.sum(aggregated_output, dim=1) + self.epsilon
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