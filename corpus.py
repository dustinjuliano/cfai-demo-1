import itertools

# Fast immutable constant for the allowed alphabet values
ALPHABET = ("A", "B")

def traditional_rle(input_list):
  """
  Accepts a list of 8 string symbols and returns an 8-element 
  alternating list of [symbol, count, symbol, count...]
  """
  output = []
  current_val = input_list[0]
  current_count = 1
  
  for val in input_list[1:]:
    if val == current_val:
      current_count += 1
    else:
      output.append(current_val)
      output.append(current_count)
      current_val = val
      current_count = 1
      
  output.append(current_val)
  output.append(current_count)
  
  if len(output) < 8:
    output += [0] * (8 - len(output))
  elif len(output) > 8:
    output = output[:8]
    
  return output

def generate_balanced_corpus(output_filename="rle_corpus.csv", duplicates_per_pattern=100):
  """
  Generates all 256 unique permutations, calculates RLE tokens,
  and saves them to disk with zero padding, no headers, and no spaces.
  """
  # Generate the entire unique universe of combinations (2^8 = 256)
  unique_inputs = list(itertools.product(ALPHABET, repeat=8))
  total_rows = 0
  
  with open(output_filename, mode="w", encoding="utf-8") as f:
    # Duplicate every single pattern equally to eliminate probability bias
    for _ in range(duplicates_per_pattern):
      for pattern in unique_inputs:
        input_list = list(pattern)
        output_list = traditional_rle(input_list)
        
        # Compress into tight, character-joined strings
        input_str = "".join(input_list)
        output_str = "".join(map(str, output_list))
        
        # Write directly to disk as raw text to avoid extra comma/quote overhead
        f.write(f"{input_str},{output_str}\n")
        total_rows += 1
        
  print(f"Successfully generated {total_rows} total rows ({total_rows / 1000:.1f} thousand lines).")
  print(f"Dataset saved to disk as '{output_filename}'")

if __name__ == "__main__":
  generate_balanced_corpus()
