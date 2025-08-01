import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Tuple

class TrainableBellMF(nn.Module):
    def __init__(self, initial_params, name="null"):
        super().__init__()

        # Initialize parameters with small values to prevent explosion
        self.a = nn.Parameter(torch.tensor([p[0] for p in initial_params], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([p[1] for p in initial_params], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor([p[2] for p in initial_params], dtype=torch.float32))

        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        membership_values = []
        epsilon = 1e-4  # Small value to avoid division by zero

        for i in range(len(self.a)):
            a_soft = torch.clamp(self.a[i], min=epsilon)  # Ensure `a_soft` is positive
            b_soft = torch.clamp(self.b[i], min=epsilon)  # Ensure `b_soft` is positive

            diff = (x - self.c[i]) / a_soft  # Avoid division by near-zero
            diff = torch.clamp(diff, min=-1e6, max=1e6)  # Limit extreme values

            membership_value = 1 / (1 + torch.pow(torch.abs(diff), 2 * b_soft))

            membership_values.append(membership_value)

        return torch.stack(membership_values)

    def show(self) -> str:

        return "BellMF Params:\n" + "\n".join(
            f"MF {i+1}: a={self.a[i].item()}, b={self.b[i].item()}, c={self.c[i].item()}"
            for i in range(len(self.a))
        )

    def visualise(self, valueRange: Tuple[float, float]):
        x = torch.linspace(valueRange[0], valueRange[1], 100)
        membership_values = self.forward(x)

        plt.figure(figsize=(10, 6))
        plt.title(self.name)
        for i, membership in enumerate(membership_values):
            plt.plot(x.numpy(), membership.detach().numpy(), label=f'Bell {i+1}')

        plt.xlabel("x")
        plt.ylabel("membership value")
        plt.legend()
        plt.grid(True)
        plt.show()


def generateBellParams(numMfs: int, valueRange: Tuple[float, float]) -> List[Tuple[float, float, float]]:

    minVal = valueRange[0]
    maxVal = valueRange[1]
    width = (maxVal - minVal) / numMfs
    centers = np.linspace(minVal, maxVal, numMfs)  # Evenly spaced centers

    params = []
    for center in centers:
        a = width * 0.6  # Width is half the step size
        b = 2.5       # A reasonable slope for bell curves
        c = center    # Center position
        params.append((a, b, c))

    return params

class TrainableTrapezoidMF(nn.Module):
    def __init__(self, initial_params):
        super().__init__()

        self.a = nn.Parameter(torch.tensor([p[0] for p in initial_params], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([p[1] for p in initial_params], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor([p[2] for p in initial_params], dtype=torch.float32))
        self.d = nn.Parameter(torch.tensor([p[3] for p in initial_params], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membership_values = []
        eps = 1e-4  # Small epsilon for numerical stability

        for i in range(len(self.a)):
            m1 = torch.clamp((x - self.a[i]) / (torch.clamp(self.b[i] - self.a[i], min=eps)), max=1.0)
            m2 = torch.clamp((self.d[i] - x) / (torch.clamp(self.d[i] - self.c[i], min=eps)), max=1.0)

            m = torch.minimum(m1, m2)
            membership_value = torch.maximum(m, torch.zeros_like(m))
            membership_values.append(membership_value)

        return torch.stack(membership_values)

    def visualise(self, valueRange: Tuple[float, float]):
        x = torch.linspace(valueRange[0], valueRange[1], 1000)
        membership_values = self.forward(x)

        plt.figure(figsize=(10, 6))
        for i, membership in enumerate(membership_values):
            plt.plot(x.numpy(), membership.detach().numpy(), label=f'Trapezoid {i+1}')

        plt.xlabel("x")
        plt.ylabel("membership value")
        plt.legend()
        plt.grid(True)
        plt.show()

def generateTrapezoidParams(numMfs: int, valueRange: Tuple[float, float]) -> List[List[float]]:

    minVal = valueRange[0]
    maxVal = valueRange[1]
    step = (maxVal - minVal) / (numMfs+1)  # Step size for overlapping regions

    params = []
    for i in range(numMfs):
        a = minVal + (step * i)  # Start of ramp-up (leftmost point)
        b = a + (step/2)    # Start of plateau (second point)
        c = b + step   # End of plateau (third point)
        d = c + (step/2)    # End of ramp-down (rightmost point)
        params.append([a, b+1.8, c-1.8, d])

    return params

def generateOutputMFParams(valueRange: Tuple[float, float]) -> List[List[float]]:

    minVal = valueRange[0]

    params = []

    # MF0
    a = minVal  # Start of ramp-up (leftmost point)
    b = minVal + 0.4  # Start of plateau (second point)
    c = -1.2  # End of plateau (third point)
    d = -0.8  # End of ramp-down (rightmost point)
    params.append([a, b, c, d])

    #MF 1 2 3
    minVal = -1.0
    maxVal = 1.0
    step = (maxVal - minVal) / (4)
    for i in range(3):
        a = minVal + (step * i)  # Start of ramp-up (leftmost point)
        b = a + (step / 2)  # Start of plateau (second point)
        c = b + step  # End of plateau (third point)
        d = c + (step / 2)  # End of ramp-down (rightmost point)
        params.append([a, b, c, d])

    #MF 4
    maxVal = valueRange[1]
    a = 0.8
    b = 1.2
    c = maxVal - 0.4
    d = maxVal
    params.append([a, b, c, d])

    return params