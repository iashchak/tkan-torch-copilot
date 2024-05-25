
import torch
from pytorch_tkan import TKAN, BSplineActivation, FixedSplineActivation

# Пример данных
inputs = torch.rand((32, 10, 8))  # batch_size=32, sequence_length=10, input_size=8
batch_size = inputs.size(0)

# Инициализация слоя
tkan_layer = TKAN(units=4, tkan_activations=[BSplineActivation(3), FixedSplineActivation(2.0)])

# Начальное состояние
initial_state = tkan_layer.cell.get_initial_state(batch_size, inputs.device)

# Прямой проход через слой
output, final_states = tkan_layer(inputs, initial_state)

# Вывод формы результата
print(output.shape)
print([state.shape for state in final_states])
