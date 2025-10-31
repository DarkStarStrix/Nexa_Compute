from datasets import load_dataset

ds = load_dataset("Allanatrix/Scientific_Research_Tokenized")
print(ds)
print(ds['train'][0])
