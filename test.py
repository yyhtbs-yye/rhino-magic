import torch
import torch.nn as nn

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyModel()

# Define the hook function
def named_forward_hook(name, module, input, output):
    print(f"Hook called on {name}")
    print(f"Input: {input}")
    print(f"Output: {output}")

from functools import partial

# Register the hook on fc1
layer_names = ['fc1', 'fc2']
for layer_name in layer_names:
    hook_handle = getattr(model, layer_name).register_forward_hook(partial(named_forward_hook, layer_name))

# Run a forward pass
x = torch.randn(1, 10)
output = model(x)

hook_handle.remove()
