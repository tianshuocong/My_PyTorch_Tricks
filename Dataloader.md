# Split the dataset

## Random 
```python
trset, valset = torch.utils.data.random_split(teset, [9000, 1000])
```

## fixed
```python
trset = torch.utils.data.Subset(teset, range(9000))
valset = torch.utils.data.Subset(teset, range(9000, 10000))
```
