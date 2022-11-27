## para name
for paras in model.state_dict():
  print(paras)
  
print(model.state_dict()['bn1.running_mean'])


## change para name
base_model = resnet18(pretrained=False).to(device)

cp_base = torch.load("./ckpt-target/resnet18-cinic.ckpt", map_location=device)['state_dict']
## ['state_dict'] (hyperparameters)

from collections import OrderedDict
new_state_dict = OrderedDict()

for key, value in cp_base.items():
    key = key[6:] # remove `model.`
    new_state_dict[key] = value

base_model.load_state_dict(new_state_dict)
base_model = base_model.to(device)


## replace BN to GN

for name, module in net.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        bn = get_layer(net, name)
        gn = nn.GroupNorm(8, bn.num_features)
        set_layer(net, name, gn)
