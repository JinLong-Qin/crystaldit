# import pandas as pd

# path = './datasets/mp_20/train.csv'
# data = pd.read_csv(path)

# print(data.columns)
# print(data.head(1))

# print(data[0]['text'])



import pickle

path = "datasets/mp_20/preprocessed/train_preprocessed_ma20.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data[0]['structure_emb'].shape)
# print(data[0]['properties_emb'].shape)



print(type(data[0]['structure_emb']))