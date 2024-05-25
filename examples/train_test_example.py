
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tkan import TKAN, BSplineActivation, FixedSplineActivation

# Пример данных
inputs = torch.rand((32, 10, 8))  # batch_size=32, sequence_length=10, input_size=8
targets = torch.randint(0, 2, (32, 4)).float()  # Пример целей для обучения

# Инициализация слоя и модели
tkan_layer = TKAN(units=4, tkan_activations=[BSplineActivation(3), FixedSplineActivation(2.0)])
model = nn.Sequential(tkan_layer, nn.Flatten(), nn.Linear(40, 4), nn.Sigmoid())  # Пример модели

# Настройка оптимизатора и функции потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Обучение модели
model.train()
for epoch in range(10):  # Обучение на 10 эпох
    optimizer.zero_grad()
    outputs, _ = tkan_layer(inputs)
    outputs = outputs[:, -1, :]  # Используем выход последнего таймстепа
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Тестирование модели
model.eval()
with torch.no_grad():
    test_inputs = torch.rand((10, 10, 8))  # Пример тестовых данных
    test_outputs, _ = tkan_layer(test_inputs)
    test_outputs = test_outputs[:, -1, :]
    print("Test Outputs:", test_outputs)
