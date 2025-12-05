import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
import inspect
import numpy as np
import time
from pathlib import Path

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

if __name__ == "__main__":

    
    # -----------------------------
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

    assert conv1_w.shape == (6,1,5,5)
    assert conv2_w.shape == (16,6,5,5)
    assert conv3_w.shape == (120,16,5,5)
    assert fc_w.shape == (10,120)

    print("Consumed all?", idx == params.size)
    print(conv1_w.shape, conv1_b.shape)
    print(conv2_w.shape, conv2_b.shape)
    print(conv3_w.shape, conv3_b.shape)
    print(fc_w.shape,   fc_b.shape)

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

    # preprocess once
    X32 = preprocess_to_32x32(images_28)        # (10000,1,32,32) float32
    # example: first item
    x0 = X32[0]
    y0 = labels[0]
    print("x0 shape:", x0.shape, "range:", float(x0.min()), float(x0.max()), "label:", int(y0))

    model = LeNet()
    model.eval()

    # -------------------------------------------
    state = model.state_dict()

    state["conv1.weight"] = torch.from_numpy(conv1_w)
    state["conv1.bias"]   = torch.from_numpy(conv1_b)

    state["conv2.weight"] = torch.from_numpy(conv2_w)
    state["conv2.bias"]   = torch.from_numpy(conv2_b)

    state["conv3.weight"] = torch.from_numpy(conv3_w)
    state["conv3.bias"]   = torch.from_numpy(conv3_b)

    state["fc.weight"]    = torch.from_numpy(fc_w)
    state["fc.bias"]      = torch.from_numpy(fc_b)

    model.load_state_dict(state)
    model.eval()
    
    print("✔ Loaded NumPy weights into PyTorch model!")

    # -------------------------------------------
    X32_torch = torch.from_numpy(X32)       # (10000,1,32,32)
    labels_torch = torch.from_numpy(labels).long()

    batch_size = 32
    N = X32_torch.shape[0]

    logits_all = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = X32_torch[i:i+batch_size]
            out = model(batch)
            logits_all.append(out)

    t1 = time.perf_counter()
    logits_all = torch.cat(logits_all, dim=0)

    preds = logits_all.argmax(dim=1)
    acc = (preds == labels_torch).float().mean().item() * 100.0

    print("========================================")
    print(f"PyTorch Inference Done!")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Total Time (PyTorch, 10k images): {t1 - t0:.3f} sec")
    print(f"Throughput: {(N / (t1 - t0)):.1f} img/sec")
    print("========================================")
    
    # -------------------------------------------
    example_input = torch.ones(1, 1, 32, 32)

    # # Compile → Torch-MLIR module
    # mlir_module = torch_mlir.compile(model, example_input, output_type="torch")

    # # Write to file
    # with open("lenet_torch.mlir", "w") as f:
    #     f.write(str(mlir_module))

    
    # print("✔ MLIR written to lenet_torch.mlir")

    # Compile → Torch-MLIR module
    mlir_module = torch_mlir.compile(model, example_input, output_type="tosa")

    # Write to file
    with open("lenet_tosa_full.mlir", "w") as f:
        f.write(str(mlir_module))

    print("✔ MLIR written to lenet_tosa_full.mlir")
