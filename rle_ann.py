import torch
import torch.nn as nn
import os

# =====================================================================
# 1. ARCHITECTURE DEFINITION (Must match the training script exactly)
# =====================================================================
class RLEModel(nn.Module):
    def __init__(self):
        super(RLEModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        
    def forward(self, x):
        return self.network(x)

# =====================================================================
# 2. INITIALIZE AND LOAD THE SAVED WEIGHTS
# =====================================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

model = RLEModel().to(device)
model_path = "rle_model.pt"

if os.path.exists(model_path):
    # Load the weights and set model to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"--> Loaded model successfully from '{model_path}' using {device.type.upper()}.\n")
else:
    print(f"CRITICAL ERROR: '{model_path}' not found in this folder.")
    print("Please run your training script first to generate the model file.")
    exit()

# =====================================================================
# 3. THE DRIVER FUNCTION (The "Neural Net" RLE algorithm)
# =====================================================================
def neural_network_rle(input_list):
    """
    Takes a Python list of 8 numbers, runs it through the trained ANN,
    applies smart rounding, and returns a standard Python list of 8 numbers.
    """
    if len(input_list) != 8:
        raise ValueError("The neural network requires an input sequence of exactly 8 numbers.")
        
    # Convert standard Python list to PyTorch tensor and add a batch dimension [1, 8]
    input_tensor = torch.tensor([input_list], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Get raw decimal prediction from the network
        raw_prediction = model(input_tensor)
        
        # Apply the smart rounding logic to get absolute integers
        rounded_prediction = torch.round(raw_prediction).int()
    
    # Strip the batch dimension, bring back to standard CPU memory, and return as a normal list
    return rounded_prediction.cpu().squeeze().tolist()


# =====================================================================
# 4. SAMPLE USAGE / DEMO
# =====================================================================
if __name__ == "__main__":
    print("--- Running Neural Network RLE Driver ---")
    
    # Test cases mirroring the data pattern your model learned: [3 of A, 2 of B, 3 of C]
    sample_input_1 = [7, 7, 7, 2, 2, 4, 4, 4]
    sample_input_2 = [1, 1, 1, 9, 9, 5, 5, 5]
    
    # Run them through the driver function
    output_1 = neural_network_rle(sample_input_1)
    output_2 = neural_network_rle(sample_input_2)
    
    print(f"Input 1 : {sample_input_1}")
    print(f"Output 1: {output_1}\n")
    
    print(f"Input 2 : {sample_input_2}")
    print(f"Output 2: {output_2}\n")
