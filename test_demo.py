import argparse
import os
import sys
from rle_py import traditional_rle
from rle_ann import neural_network_rle

# Fast, immutable constant for the allowed alphabet values
ALPHABET = ("A", "B")

def generate_constrained_sequence():
  """
  Generates an 8-element sequence using os.urandom.
  Lengths are dynamically clamped to fit the remaining space.
  """
  seq = []
  while len(seq) < 8:
    remaining_space = 8 - len(seq)
    symbol_byte = os.urandom(1)[0]
    symbol = ALPHABET[symbol_byte % len(ALPHABET)]
    length_byte = os.urandom(1)[0]
    run_length = 1 + (length_byte % remaining_space)
    seq.extend([symbol] * run_length)
  return seq

def main():
  parser = argparse.ArgumentParser(description="RLE Module Tester")
  parser.add_argument(
    "-m", "--module",
    choices=["rle_py", "rle_ann", "both"],
    default="both",
    help="The module framework to test (default: both)"
  )
  parser.add_argument(
    "-r", "--runs",
    type=int,
    default=1000,
    help="Number of test iterations to execute (default: 1000)"
  )
  parser.add_argument(
    "-c", "--corpus",
    type=str,
    default=None,
    help="Path to a corpus CSV file to run tests against instead of generating data on the fly"
  )
  parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="Show detailed output for each individual test case"
  )

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(0)

  args = parser.parse_args()
  total_runs = args.runs
  py_success = 0
  ann_success = 0
  both_match = 0

  # 1. Acquire Input Data Source
  input_sequences = []
  if args.corpus:
    if not os.path.exists(args.corpus):
      print(f"Error: Corpus file '{args.corpus}' not found.")
      sys.exit(1)
    with open(args.corpus, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line and "," in line:
          input_str, _ = line.split(",")
          input_sequences.append(list(input_str))

    # Cap total runs if corpus size is smaller than requested iterations
    if len(input_sequences) < total_runs:
      total_runs = len(input_sequences)
    input_sequences = input_sequences[:total_runs]
    source_msg = f"Corpus File '{args.corpus}'"
  else:
    for _ in range(total_runs):
      input_sequences.append(generate_constrained_sequence())
    source_msg = "Dynamic On-The-Fly Generation"

  print("=" * 60)
  print(f"Executing {total_runs} Test Cases via {source_msg}...")
  print("=" * 60)

  # 2. Evaluate Sequences
  for input_seq in input_sequences:
    ground_truth = traditional_rle(input_seq)
    py_res = None
    ann_res = None

    if args.module in ["rle_py", "both"]:
      py_res = traditional_rle(input_seq)
      if py_res == ground_truth:
        py_success += 1

    if args.module in ["rle_ann", "both"]:
      try:
        ann_res = neural_network_rle(input_seq)
        if ann_res == ground_truth:
          ann_success += 1
      except Exception:
        ann_res = "ERROR"

    if args.module == "both":
      if py_res == ann_res and ann_res != "ERROR":
        both_match += 1

    if args.verbose:
      if args.module == "rle_py":
        passed = (py_res == ground_truth)
        out_str = " ".join(map(str, py_res))
      elif args.module == "rle_ann":
        passed = (ann_res == ground_truth)
        out_str = " ".join(map(str, ann_res)) if isinstance(ann_res, list) else str(ann_res)
      else:
        passed = (py_res == ground_truth and ann_res == ground_truth)
        py_str = " ".join(map(str, py_res))
        ann_str = " ".join(map(str, ann_res)) if isinstance(ann_res, list) else str(ann_res)
        out_str = f"PY: {py_str} | ANN: {ann_str}"

      chk = "[x]" if passed else "[ ]"
      in_str = " ".join(map(str, input_seq))
      print(f"{chk} {in_str}  ->  {out_str}")

  print("=" * 40)
  print("         RLE TESTER STATISTICS")
  print("=" * 40)
  print(f"Target Module: {args.module}")
  print(f"Data Source:   {source_msg}")
  print(f"Total Runs:    {total_runs}")
  print("-" * 40)
  if args.module == "rle_py":
    print(f"Python Accuracy: {(py_success / total_runs) * 100:.2f}% ({py_success}/{total_runs})")
  elif args.module == "rle_ann":
    print(f"Neural Net Accuracy: {(ann_success / total_runs) * 100:.2f}% ({ann_success}/{total_runs})")
  elif args.module == "both":
    print(f"Python Accuracy:     {(py_success / total_runs) * 100:.2f}% ({py_success}/{total_runs})")
    print(f"Neural Net Accuracy: {(ann_success / total_runs) * 100:.2f}% ({ann_success}/{total_runs})")
    print(f"Identical Outputs:   {(both_match / total_runs) * 100:.2f}% ({both_match}/{total_runs})")
  print("=" * 40)

if __name__ == "__main__":
  main()
