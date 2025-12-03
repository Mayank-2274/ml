import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
df = fetch_california_housing(as_frame=True).frame

# Correlation matrix
corr = df.corr()

# Heatmap visualization
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
