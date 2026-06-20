from rle_ann import neural_network_rle
from rle_py import traditional_rle

def run_comparison(test_case, case_number):
  print(f"Test Case {case_number}")
  print(f"  Input:      {test_case}")

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

  case_1 = ["A", "A", "A", "B", "B", "A", "A", "A"]
  case_2 = ["B", "B", "B", "B", "A", "A", "B", "B"]

  run_comparison(case_1, 1)
  run_comparison(case_2, 2)
