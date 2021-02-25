import torch

def outputFunc(logits, output_func_name):  
  """
  Calculates the LogSoftmax or Softmax output

  Argument: 
    logits (torch.tensor, requires_grad=True): return value of "getLogits" function from getLogits.py
    func (str):  'logsoftmax', 'softmax'
  
  Returns: None
    function prints output values for given softmax or logsoftmax function
  """
  if output_func_name == 'logsoftmax':
    output = torch.nn.LogSoftmax(dim=1)(logits).tolist()[0]
  elif output_func_name == 'softmax':
    output = torch.nn.Softmax(dim=1)(logits).tolist()[0]
  else:
    print("Something went wrong in outputFunction.py")


  print("")
  print("nn." + output_func_name + " Distribution >>>")
  print("---------------")
  print(f"Normal: {output[0]}", 
        f"Non-Covid Pneumonia: {output[1]}",
        f"Covid Pneumonia: {output[2]}", sep='\n')
  print("---------------")
  print("")

  return None