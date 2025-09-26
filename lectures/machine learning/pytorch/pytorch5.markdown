# PyTorch Learning Notes part 5
## work with module glob for reading data

Example without glob
```
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

filename=('/content/S001.txt')
df= pd.read_csv(filename,sep='\t',header=None)
sig= df.iloc[:,0].to_numpy()
plt.subplot(2,1,1)
plt.plot(sig)
plt.show
```
* Note: for change a list of arrays to tensor , first cconvert list to to array then array to tensor

