import torch
import torch.nn as nn
import torch.optim as optim
import os

# Hardware detection
device = torch.device(
  "cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu"
)
print(f"--> Training on device: {device.type.upper()}\n")

ALPHABET = ("A", "B")

def generate_training_data(num_samples=30000):
  X_list = []
  Y_list = []

  for _ in range(num_samples):
    seq = []
    while len(seq) < 8:
      remaining_space = 8 - len(seq)
      symbol_byte = os.urandom(1)[0]
      symbol = ALPHABET[symbol_byte % len(ALPHABET)]
      length_byte = os.urandom(1)[0]
      run_length = 1 + (length_byte % remaining_space)
      seq.extend([symbol] * run_length)

    # Convert input symbols to One-Hot Encoding
    one_hot_input = []
    for sym in seq:
      if sym == "A":
        one_hot_input.extend([1.0, 0.0])
      else:
        one_hot_input.extend([0.0, 1.0])

    # Calculate pure RLE pairs
    rle_pairs = []
    current_val = seq[0]
    current_count = 1
    for val in seq[1:]:
      if val == current_val:
        current_count += 1
      else:
        rle_pairs.append(current_val)
        rle_pairs.append(current_count)
        current_val = val
        current_count = 1
    rle_pairs.append(current_val)
    rle_pairs.append(current_count)

    # Pad or truncate to exactly 8 elements
    if len(rle_pairs) < 8:
      rle_pairs += [0] * (8 - len(rle_pairs))
    else:
      rle_pairs = rle_pairs[:8]

    # Map tokens to 11 classification categories
    target_indices = []
    for i, token in enumerate(rle_pairs):
      if token == 0:
        target_indices.append(0)
      elif i % 2 == 0:
        target_indices.append(1 if token == "A" else 2)
      else:
        target_indices.append(2 + int(token))

    X_list.append(one_hot_input)
    Y_list.append(target_indices)

  return torch.tensor(X_list, dtype=torch.float32), torch.tensor(Y_list, dtype=torch.long)

X_train, Y_train = generate_training_data(30000)

class RLEClassificationModel(nn.Module):
  def __init__(self):
    super(RLEClassificationModel, self).__init__()
    self.network = nn.Sequential(
      nn.Linear(16, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 88)
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
