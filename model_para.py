
for paras in model.state_dict():
  print(paras)
  
  
print(model.state_dict()['bn1.running_mean'])
