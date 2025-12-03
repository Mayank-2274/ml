import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
num = df.select_dtypes('number')

# Histograms
num.hist(figsize=(10,6))
plt.tight_layout(); plt.show()

# Boxplots
plt.figure(figsize=(10,6))
sns.boxplot(data=num)
plt.xticks(rotation=90)
plt.show()

# Outlier Detection (IQR)
for c in num:
    Q1, Q3 = num[c].quantile([0.25,0.75])
    IQR = Q3 - Q1
    outliers = num[(num[c] < Q1-1.5*IQR) | (num[c] > Q3+1.5*IQR)]
    print(f"{c}: {len(outliers)} outliers")
