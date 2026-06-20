from rle_ann import neural_network_rle
from rle_py import traditional_rle

def run_comparison(test_case, case_number):
  print(f"Test Case {case_number}")
  print(f"  Input:      {test_case}")

  # Execute both models
  py_result = traditional_rle(test_case)
  nn_result = neural_network_rle(test_case)

  match = "PASS" if py_result == nn_result else "FAIL"

  print(f"  Python RLE: {py_result}")
  print(f"  Neural Net: {nn_result}")
  print(f"  Status:     {match}")
  print("-" * 40)

if __name__ == "__main__":
  print("=" * 40)
  print("   RLE Twin-Program Comparison Demo")
  print("=" * 40)
  print()

  # Standard patterns that match the training data
  good_case_1 = [8, 8, 8, 3, 3, 9, 9, 9]
  good_case_2 = [1, 1, 1, 5, 5, 2, 2, 2]

  # Anomalous case that breaks the training pattern
  chaotic_case = [1, 2, 3, 4, 5, 6, 7, 8]

  run_comparison(good_case_1, 1)
  run_comparison(good_case_2, 2)

  print("Testing untrained chaotic pattern to observe ANN limitations:")
  print("-" * 40)
  run_comparison(chaotic_case, 3)
