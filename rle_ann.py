import torch
import torch.nn as nn
import os

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

device = torch.device(
  "cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu"
)

model = RLEClassificationModel().to(device)
model_path = "rle_model.pt"

if os.path.exists(model_path):
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()
else:
  print(f"Error: {model_path} missing. Run train.py first.")

def neural_network_rle(input_list):
  one_hot_input = []
  for sym in input_list:
    if sym == "A":
      one_hot_input.extend([1.0, 0.0])
    else:
      one_hot_input.extend([0.0, 1.0])

  input_tensor = torch.tensor([one_hot_input], dtype=torch.float32).to(device)

  with torch.no_grad():
    outputs = model(input_tensor)
    class_predictions = torch.argmax(outputs, dim=2).squeeze().tolist()

  final_output = []
  for i, class_idx in enumerate(class_predictions):
    if class_idx == 0:
      final_output.append(0)
    elif class_idx == 1:
      final_output.append("A")
    elif class_idx == 2:
      final_output.append("B")
    else:
      final_output.append(class_idx - 2)

  return final_output
