
import torch
from pytorch_tkan.spline import FixedSplineActivation, PowerSplineActivation, BSplineActivation

def test_fixed_spline_activation():
    activation = FixedSplineActivation(exponent=2.0)
    inputs = torch.tensor([0.5, 1.0, 1.5])
    outputs = activation(inputs)
    expected_outputs = torch.tensor([0.25, 1.0, 1.0])
    assert torch.allclose(outputs, expected_outputs), f"Expected {expected_outputs}, but got {outputs}"

def test_power_spline_activation():
    activation = PowerSplineActivation(initial_exponent=2.0)
    inputs = torch.tensor([0.5, 1.0, 1.5])
    outputs = activation(inputs)
    expected_outputs = torch.tensor([0.25, 1.0, 2.25])
    assert torch.allclose(outputs, expected_outputs), f"Expected {expected_outputs}, but got {outputs}"

def test_bspline_activation():
    activation = BSplineActivation(num_bases=3, order=2)
    inputs = torch.tensor([0.5, 1.0])
    outputs = activation(inputs)
    assert outputs.shape == inputs.shape, f"Expected shape {inputs.shape}, but got {outputs.shape}"
