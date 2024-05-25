
# PyTorch TKAN

This project contains a PyTorch implementation of the Temporal Kolmogorov-Arnold Network (TKAN) with custom spline activation functions.

## Installation

To install the dependencies, you can use [Poetry](https://python-poetry.org/). If you don't have Poetry installed, follow the [installation guide](https://python-poetry.org/docs/#installation).

```bash
poetry install
```

## Usage

### Example Usage

```python
import torch
from pytorch_tkan import TKAN, BSplineActivation, FixedSplineActivation

# Example data
inputs = torch.rand((32, 10, 8))  # batch_size=32, sequence_length=10, input_size=8
batch_size = inputs.size(0)

# Initialize layer
tkan_layer = TKAN(units=4, tkan_activations=[BSplineActivation(3), FixedSplineActivation(2.0)])

# Initial state
initial_state = tkan_layer.cell.get_initial_state(batch_size, inputs.device)

# Forward pass through the layer
output, final_states = tkan_layer(inputs, initial_state)

# Output shapes
print(output.shape)
print([state.shape for state in final_states])
```

### Example Training and Testing

```python
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tkan import TKAN, BSplineActivation, FixedSplineActivation

# Example data
inputs = torch.rand((32, 10, 8))  # batch_size=32, sequence_length=10, input_size=8
targets = torch.randint(0, 2, (32, 4)).float()  # Example targets for training

# Initialize layer and model
tkan_layer = TKAN(units=4, tkan_activations=[BSplineActivation(3), FixedSplineActivation(2.0)])
model = nn.Sequential(tkan_layer, nn.Flatten(), nn.Linear(40, 4), nn.Sigmoid())  # Example model

# Setup optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Train the model
model.train()
for epoch in range(10):  # Train for 10 epochs
    optimizer.zero_grad()
    outputs, _ = tkan_layer(inputs)
    outputs = outputs[:, -1, :]  # Use the output of the last timestep
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Test the model
model.eval()
with torch.no_grad():
    test_inputs = torch.rand((10, 10, 8))  # Example test data
    test_outputs, _ = tkan_layer(test_inputs)
    test_outputs = test_outputs[:, -1, :]
    print("Test Outputs:", test_outputs)
```

## Running Tests

To run the tests, use `pytest`:

```bash
pytest
```

## Linting

This project uses `flake8` for linting. To run the linter, use:

```bash
flake8
```
