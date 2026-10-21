"""Compact conditional networks for scalar residual distributions."""
import math
import torch
from torch import nn


class ConditionEncoder(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class GaussianResidual(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.encoder = ConditionEncoder(inputs)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        result = self.head(self.encoder(x))
        return result[:, :1], result[:, 1:].clamp(-7., 3.)


class ResidualDenoiser(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.encoder = ConditionEncoder(inputs)
        self.register_buffer('frequencies', torch.exp(torch.linspace(0., math.log(1000.), 16)))
        self.layers = nn.Sequential(nn.Linear(97, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))

    def denoise(self, z, step, condition):
        phase = step[:, None]*self.frequencies[None, :]*2*math.pi
        embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return self.layers(torch.cat([z, embedding, condition], dim=1))

    def forward(self, z, step, x):
        return self.denoise(z, step, self.encoder(x))
