import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (1, 32, 32)
        # C1: 1 -> 6, kernel 5, output: (6, 28, 28)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)

        # S2: MaxPool 2x2 -> (6, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # C3: 6 -> 16, kernel 5, output: (16, 10, 10)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # S4: MP 2x2 -> (16, 5, 5)
        # Already using same pool layer

        # C5: 16 -> 120, kernel 5, out = (120, 1, 1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        # F6: 120 -> 10 (logits)
        self.fc = nn.Linear(120, 10)

    def forward(self, x):
        # C1 → ReLU → MP
        x = self.pool(F.relu(self.conv1(x)))      # (N,6,14,14)

        # C3 → ReLU → MP
        x = self.pool(F.relu(self.conv2(x)))      # (N,16,5,5)

        # C5 → ReLU
        x = F.relu(self.conv3(x))                 # (N,120,1,1)

        # Flatten 120
        x = torch.flatten(x, 1)                   # (N,120)

        # Final FC → logits
        x = self.fc(x)                            # (N,10)
        return x

model = LeNet()
model.eval()

example_input = torch.ones(1, 1, 32, 32)
mlir_module = torch_mlir.compile(model, example_input, output_type="tosa")

# Write to file
with open("lenet_tosa.mlir", "w") as f:
    f.write(str(mlir_module))

print("✔ MLIR written to lenet_tosa.mlir")