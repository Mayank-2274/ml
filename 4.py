import pandas as pd

df = pd.read_csv("a.csv")

data = df.values

target_col = -1

num_features = data.shape[1] - 1
hypothesis = ["Ø"] * num_features   

# FIND-S algorithm
for row in data:
    if row[target_col].lower() == "yes" or row[target_col].lower() == "positive":
        if hypothesis[0] == "Ø":  
            hypothesis = row[:-1].tolist()
        else:
            for i in range(num_features):
                if hypothesis[i] != row[i]:
                    hypothesis[i] = "?"   

print("Final Hypothesis using FIND-S:")
print(hypothesis)
