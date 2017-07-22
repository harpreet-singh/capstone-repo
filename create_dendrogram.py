import pandas as pd
import numpy as np

from scipy import stats
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import datetime


dft = pd.read_csv('snp500.csv', index_col = 0)

#dft = df.iloc[-500:]
print("The shape of the dataset: {}".format(dft.shape))
dft.dropna( inplace=True)

print("Calculating means...")
for c in dft.columns:
    dft[c] = dft[c]/dft[c].mean()
print(dft.head())

Z = hac.linkage(dft, 'single', 'correlation')

# Plot the dendogram

# Used HAC to get clustered stock tickers. 
# Chose tickers from same group to make use of know informations. New affects particular sectors together.
# Did further analysis on these to predict the price.

fig =plt.gcf()

plt.title('HAC Dendrogram')
plt.xlabel('Tickers')
plt.ylabel('distance')
hac.dendrogram(
    Z,
   
    leaf_rotation=90.,  
    leaf_font_size=12.,  
    labels=list(dft.index)
)
fig.set_size_inches(100, 60)

fig.savefig('cluster_dendo.png', dpi=100)
print("Plotting finished")

