def traditional_rle(input_list):
  """
  Accepts a list of 8 string symbols and returns an 8-element
  alternating list of [symbol, count, symbol, count...]
  """
  if len(input_list) != 8:
    raise ValueError("Input list must contain exactly 8 elements.")

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
