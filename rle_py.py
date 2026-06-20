def traditional_rle(input_list):
  """
  Pure Python algorithmic execution of Run-Length Encoding.
  Converts a list of 8 numbers into an RLE pair list padded/truncated to 8 elements.
  """
  if len(input_list) != 8:
    raise ValueError("Input list must contain exactly 8 numbers.")

  output = []
  current_val = input_list[0]
  current_count = 1

  # Loop through the list starting from the second element
  for val in input_list[1:]:
    if val == current_val:
      current_count += 1
    else:
      output.append(current_val)
      output.append(current_count)
      current_val = val
      current_count = 1

  # Append the final sequence group
  output.append(current_val)
  output.append(current_count)

  # Enforce strict 8-number output constraint to match the NN shape
  if len(output) < 8:
    # Pad with trailing zeros if code is shorter than 8 elements
    output += [0] * (8 - len(output))
  elif len(output) > 8:
    # Truncate if the sequence is highly chaotic and exceeds 8 elements
    output = output[:8]

  return output
