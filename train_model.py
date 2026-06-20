import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =====================================================================
# 1. HARDWARE ACCELERATION SETUP
# =====================================================================
device = torch.device(
  "cuda" if torch.cuda.is_available()              # NVIDIA GPU (Windows/Linux/Colab)
  else "mps" if torch.backends.mps.is_available()  # Apple Silicon (Mac M1/M2/M3)
  else "cpu"                                       # Standard Processor fallback
)
print(f"--> Using hardware accelerator device: {device.type.upper()}\n")


# =====================================================================
# 2. GENERATE SYNTHETIC DATA
# =====================================================================
# Input: [A, A, A, B, B, C, C, C] -> Output: [A, 3, B, 2, C, 3, 0, 0]
def generate_mock_rle_data(num_samples=1000):
  X = []
  Y = []
  for _ in range(num_samples):
    a, b, c = np.random.randint(1, 10, size=3)
    while b == a: b = np.random.randint(1, 10)
    while c == b: c = np.random.randint(1, 10)

    input_seq = [a, a, a, b, b, c, c, c]
    output_seq = [a, 3, b, 2, c, 3, 0, 0]

    X.append(input_seq)
    Y.append(output_seq)

  return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Generate data arrays
X_train, Y_train = generate_mock_rle_data(2000)
X_test, Y_test = generate_mock_rle_data(5)


# =====================================================================
# 3. DEFINE THE ANN ARCHITECTURE
# =====================================================================
class RLEModel(nn.Module):
  def __init__(self):
    super(RLEModel, self).__init__()
    self.network = nn.Sequential(
      nn.Linear(8, 128),   # Input Layer (8 nodes) -> Hidden Layer
      nn.ReLU(),
      nn.Linear(128, 64),  # Hidden Layer
      nn.ReLU(),
      nn.Linear(64, 8)     # Output Layer (8 nodes)
    )

  def forward(self, x):
    return self.network(x)


# =====================================================================
# 4. INITIALIZE MODEL, LOSS, AND OPTIMIZER
# =====================================================================
# Create the model and immediately push its weights onto the chosen hardware device
model = RLEModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


# =====================================================================
# 5. TRAINING LOOP
# =====================================================================
print("--- Training Started ---")
epochs = 500
batch_size = 32

for epoch in range(epochs):
  model.train()

  # Shuffle dataset indexes every epoch
  permutation = torch.randperm(X_train.size()[0])

  epoch_loss = 0.0
  for i in range(0, X_train.size()[0], batch_size):
    indices = permutation[i:i+batch_size]

    # Move the data batch to the same hardware device as the model
    batch_x = X_train[indices].to(device)
    batch_y = Y_train[indices].to(device)

    # Forward pass
    predictions = model(batch_x)
    loss = criterion(predictions, batch_y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()

  if (epoch + 1) % 50 == 0:
    avg_loss = epoch_loss / (X_train.size()[0] / batch_size)
    print(f"Epoch {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f}")

print("--- Training Complete ---")

# SAVE MODEL WEIGHTS HERE
torch.save(model.state_dict(), "rle_model.pt")
print("--> Model successfully saved locally as 'rle_model.pt'\n")


# =====================================================================
# 6. EVALUATION AND TESTING WITH ROUNDING
# =====================================================================
model.eval()
with torch.no_grad():
  # Push the test samples to the hardware device
  test_inputs = X_test.to(device)

  # Run predictions through the network
  raw_predictions = model(test_inputs)

  # Smart rounding: snapshot floating decimals to closest integers
  rounded_predictions = torch.round(raw_predictions).int()

  # Move results back to CPU memory just for clean print formatting
  X_test_cpu = X_test.int().tolist()
  Y_test_cpu = Y_test.int().tolist()
  raw_preds_cpu = raw_predictions.cpu().tolist()
  rounded_preds_cpu = rounded_predictions.cpu().tolist()

  print("--- Test Results ---")
  for i in range(len(X_test)):
    print(f"Test Sample {i+1}:")
    print(f"  Input Sequence : {X_test_cpu[i]}")
    print(f"  Target RLE     : {Y_test_cpu[i]}")
    print(f"  Raw NN Output  : {[round(num, 2) for num in raw_preds_cpu[i]]}")
    print(f"  Rounded Output : {rounded_preds_cpu[i]}")
    print("-" * 30)
