import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
import numpy as np

from pathlib import Path

#%%
# Open LeNet Weights
path = "data/params.bin"
params = np.fromfile(path, dtype=np.float32)
print("Total floats:", params.size)  # expect 51902

idx = 0

# C1: 6×1×5×5 + 6b
conv1_w = params[idx : idx + 6*1*5*5].reshape(6, 1, 5, 5); idx += 6*1*5*5
conv1_b = params[idx : idx + 6];                            idx += 6

# C3: 16×6×5×5 + 16b
conv2_w = params[idx : idx + 16*6*5*5].reshape(16, 6, 5, 5); idx += 16*6*5*5
conv2_b = params[idx : idx + 16];                            idx += 16

# C5: 120×16×5×5 + 120b   (convolution producing 120×1×1; often “looks like” FC)
conv3_w = params[idx : idx + 120*16*5*5].reshape(120, 16, 5, 5); idx += 120*16*5*5
conv3_b = params[idx : idx + 120];                              idx += 120

# Final FC: 10×120 + 10b
fc_w = params[idx : idx + 10*120].reshape(10, 120); idx += 10*120
fc_b = params[idx : idx + 10];                      idx += 10

print("Consumed all?", idx == params.size)
print(conv1_w.shape, conv1_b.shape)
print(conv2_w.shape, conv2_b.shape)
print(conv3_w.shape, conv3_b.shape)
print(fc_w.shape,   fc_b.shape)
#%%
# Open Test data
# this code block process labels and images
labels_path = Path("data/labels.bin")
images_path = Path("data/images.bin")

# labels: skip 8-byte header (magic, count) 
with open(labels_path, "rb") as f:
    magic, num_labels = np.frombuffer(f.read(8), dtype=">u4")
    labels = np.frombuffer(f.read(), dtype=np.uint8)

print(f"labels: magic={magic} count={num_labels} shape={labels.shape}")
assert labels.shape[0] == num_labels

# images: skip 16-byte header (magic, count, rows, cols) 
with open(images_path, "rb") as f:
    magic_i, num_images, rows, cols = np.frombuffer(f.read(16), dtype=">u4")
    pixels = np.frombuffer(f.read(), dtype=np.uint8)

print(f"images header: magic={magic_i} count={num_images} rows={rows} cols={cols}")
assert num_images == num_labels == 10000
assert rows == 28 and cols == 28
assert pixels.size == num_images * rows * cols

images_28 = pixels.reshape(num_images, rows, cols)
print("images_28 shape:", images_28.shape)

# --------- 1) preprocess once: (N,28,28) -> (N,1,32,32) ---------
def preprocess_to_32x32(images_28: np.ndarray) -> np.ndarray:
    """
    images_28: (N,28,28) uint8
    returns:   (N,1,32,32) float32, center scaled to [-1,1], 2px border = -1
    """
    N = images_28.shape[0]
    X32 = np.full((N, 1, 32, 32), -1.0, dtype=np.float32)
    core = (images_28.astype(np.float32) / 255.0) * 2.0 - 1.0
    X32[:, 0, 2:30, 2:30] = core
    return X32

X32 = preprocess_to_32x32(images_28)        # (10000,1,32,32) float32

#%%

# Fix seeds so weight init and outputs are repeatable.
torch.manual_seed(0)
np.random.seed(0)

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

with torch.no_grad():
    model.conv1.weight.copy_(torch.from_numpy(conv1_w))
    model.conv1.bias.copy_(torch.from_numpy(conv1_b))

    model.conv2.weight.copy_(torch.from_numpy(conv2_w))
    model.conv2.bias.copy_(torch.from_numpy(conv2_b))

    model.conv3.weight.copy_(torch.from_numpy(conv3_w))
    model.conv3.bias.copy_(torch.from_numpy(conv3_b))

    model.fc.weight.copy_(torch.from_numpy(fc_w))
    model.fc.bias.copy_(torch.from_numpy(fc_b))

example_input = torch.ones(1, 1, 32, 32)
test_input = torch.from_numpy(X32)
mlir_module = torch_mlir.compile(model, example_input, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

with open("lenet_torch_bm.mlir", "w") as f:
    f.write(str(mlir_module))

print("✔ MLIR written to lenet_torch_bm.mlir")
