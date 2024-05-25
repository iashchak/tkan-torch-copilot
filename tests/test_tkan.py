
import torch
from pytorch_tkan import TKAN, BSplineActivation, FixedSplineActivation

def test_tkan():
    inputs = torch.rand((32, 10, 8))
    tkan_layer = TKAN(units=4, tkan_activations=[BSplineActivation(3), FixedSplineActivation(2.0)])
    initial_state = tkan_layer.cell.get_initial_state(32, inputs.device)
    output, final_states = tkan_layer(inputs, initial_state)
    assert output.shape == (32, 10, 4), f"Expected shape (32, 10, 4), but got {output.shape}"
    assert len(final_states) == 2 + len(tkan_layer.cell.tkan_sub_layers), "Mismatch in the number of final states"
