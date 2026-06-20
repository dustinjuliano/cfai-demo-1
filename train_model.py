# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Hardware detection
device = torch.device(
  "cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu"
)
print(f"--> Training on device: {device.type.upper()}\n")

def load_balanced_corpus(filename="rle_corpus.csv"):
  """
  Reads the space-optimized CSV corpus from disk, converts strings
  into numerical features, and builds the training tensors.
  """
  if not os.path.exists(filename):
    print(f"Error: {filename} not found. Please run corpus.py first.")
    sys.exit(1)

  X_list = []
  Y_list = []

  print(f"Loading dataset from '{filename}'...")
  with open(filename, mode="r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line or "," not in line:
        continue

      input_str, output_str = line.split(",")

      # 1. Rebuild One-Hot encoded input vector from the packed symbols
      one_hot_input = []
      for sym in input_str:
        if sym == "A":
          one_hot_input.extend([1.0, 0.0])
        else:
          one_hot_input.extend([0.0, 1.0])

      # 2. Rebuild the 11-class indices from the packed output string
      target_indices = []
      for char in output_str:
        if char == "0":
          target_indices.append(0)
        elif char == "A":
          target_indices.append(1)
        elif char == "B":
          target_indices.append(2)
        else:
          # Converts character digits '1'-'8' to class indices 3-10
          target_indices.append(2 + int(char))

      X_list.append(one_hot_input)
      Y_list.append(target_indices)

  print(f"Successfully loaded {len(X_list)} samples into memory.")
  return torch.tensor(X_list, dtype=torch.float32), torch.tensor(Y_list, dtype=torch.long)

# Load the perfectly balanced corpus from disk
X_train, Y_train = load_balanced_corpus("rle_corpus.csv")

class RLEClassificationModel(nn.Module):
  def __init__(self):
    super(RLEClassificationModel, self).__init__()
    self.network = nn.Sequential(
      nn.Linear(16, 256),  # 16 inputs (8 symbols * 2 one-hot features)
      nn.ReLU(),
      nn.Linear(256, 256), # Expanded hidden layers for perfect boundary mapping
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 88)   # Output: 8 slots * 11 classes per slot = 88 units
    )

  def forward(self, x):
    out = self.network(x)
    return out.view(-1, 8, 11)

model = RLEClassificationModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200
batch_size = 64

print("--- Training Classification Model Started ---")
for epoch in range(epochs):
  model.train()
  permutation = torch.randperm(X_train.size()[0])
  epoch_loss = 0.0

  for i in range(0, X_train.size()[0], batch_size):
    indices = permutation[i:i+batch_size]
    batch_x = X_train[indices].to(device)
    batch_y = Y_train[indices].to(device)

    predictions = model(batch_x)

    # Flatten shapes for CrossEntropy tracking evaluation
    loss = criterion(predictions.view(-1, 11), batch_y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()

  if (epoch + 1) % 20 == 0:
    avg_loss = epoch_loss / (X_train.size()[0] / batch_size)
    print(f"Epoch {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f}")

print("--- Training Complete ---")
torch.save(model.state_dict(), "rle_model.pt")
print("--> Model saved successfully as 'rle_model.pt'")
