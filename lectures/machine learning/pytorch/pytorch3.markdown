# PyTorch Learning Notes part 3

## torch.split

torch.split() is a powerful function in PyTorch that allows you to divide a single tensor into multiple, smaller tensors. This is essentially the reverse operation of torch.cat() and is incredibly useful when you need to process different parts of a tensor separately.

The function takes two main arguments:

* The input tensor that you want to split.

* The split size or a list of sizes. This tells the function how to perform the division. You can provide a single integer to split the tensor into chunks of equal size, or you can provide a list of integers to specify the size of each chunk.

* The dimension (dim) along which to perform the split.

```
# This script demonstrates how to use torch.split() to divide a tensor.

import torch

# --- Example 1: Splitting a 1D Tensor into equal-sized chunks ---
print("--- Example 1: Splitting a 1D Tensor ---")

# Create a 1D tensor with 12 elements
tensor_1d = torch.arange(12)
print(f"Original 1D Tensor:\n{tensor_1d}")
print(f"Original Shape: {tensor_1d.shape}\n")

# Split the tensor into 3 equal parts (chunks of size 4)
split_tensors_equal = torch.split(tensor_1d, split_size_or_sections=4)
print(f"Splitting into equal-sized chunks (size=4):")
for i, t in enumerate(split_tensors_equal):
    print(f"  Chunk {i+1}: {t}")
print(f"Number of resulting tensors: {len(split_tensors_equal)}\n")

# Split the tensor using a list of sizes
split_tensors_list = torch.split(tensor_1d, split_size_or_sections=[2, 5, 5])
print(f"Splitting with a list of sizes ([2, 5, 5]):")
for i, t in enumerate(split_tensors_list):
    print(f"  Chunk {i+1}: {t}")
print("-" * 30)

# --- Example 2: Splitting a 2D Tensor along a dimension ---
print("--- Example 2: Splitting a 2D Tensor ---")

# Create a 2D tensor (4x6)
tensor_2d = torch.arange(24).reshape(4, 6)
print(f"Original 2D Tensor (4x6):\n{tensor_2d}\n")

# Split along dimension 0 (the row dimension) into 2 equal-sized chunks
split_rows = torch.split(tensor_2d, split_size_or_sections=2, dim=0)
print("Splitting along dim=0 (rows) into 2 chunks of size 2:")
print(f"  Chunk 1:\n{split_rows[0]}")
print(f"  Chunk 2:\n{split_rows[1]}\n")

# Split along dimension 1 (the column dimension) into chunks of size 3
split_cols = torch.split(tensor_2d, split_size_or_sections=3, dim=1)
print("Splitting along dim=1 (columns) into 2 chunks of size 3:")
print(f"  Chunk 1:\n{split_cols[0]}")
print(f"  Chunk 2:\n{split_cols[1]}")
```
### output
```
--- Example 1: Splitting a 1D Tensor ---
Original 1D Tensor:
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
Original Shape: torch.Size([12])

Splitting into equal-sized chunks (size=4):
  Chunk 1: tensor([0, 1, 2, 3])
  Chunk 2: tensor([4, 5, 6, 7])
  Chunk 3: tensor([ 8,  9, 10, 11])
Number of resulting tensors: 3

Splitting with a list of sizes ([2, 5, 5]):
  Chunk 1: tensor([0, 1])
  Chunk 2: tensor([2, 3, 4, 5, 6])
  Chunk 3: tensor([ 7,  8,  9, 10, 11])
------------------------------
--- Example 2: Splitting a 2D Tensor ---
Original 2D Tensor (4x6):
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])

Splitting along dim=0 (rows) into 2 chunks of size 2:
  Chunk 1:
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
  Chunk 2:
tensor([[12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])

Splitting along dim=1 (columns) into 2 chunks of size 3:
  Chunk 1:
tensor([[ 0,  1,  2],
        [ 6,  7,  8],
        [12, 13, 14],
        [18, 19, 20]])
  Chunk 2:
tensor([[ 3,  4,  5],
        [ 9, 10, 11],
        [15, 16, 17],
        [21, 22, 23]])
```


# work with Data

## read data with numpy

```
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
```
## read data and make dataframe with panda
```
import pandas as pd

#dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

dic= {'length':[12,15,11,14,9,13],
      'width':[5,8,7,8,4,10],
      'intensity':[155,88,210,154,160,145],
      'label':[1,1,1,2,2,2]
}

df= pd.DataFrame(dic)
print(df)
print(df.head(3))
print(df.tail(2))

print(df.columns)

col=df.columns
lenV=df.loc[:,col[0]].values
print(lenV)

wV=df.loc[:,col[1]].values
print(wV)

inV=df.loc[:,col[2]].values
print(inV)
```
### output
```
   length  width  intensity  label
0      12      5        155      1
1      15      8         88      1
2      11      7        210      1
3      14      8        154      2
4       9      4        160      2
5      13     10        145      2
   length  width  intensity  label
0      12      5        155      1
1      15      8         88      1
2      11      7        210      1
   length  width  intensity  label
4       9      4        160      2
5      13     10        145      2
Index(['length', 'width', 'intensity', 'label'], dtype='object')
[12 15 11 14  9 13]
[ 5  8  7  8  4 10]
[155  88 210 154 160 145]
```
now convert df to numpy array
```
a=df.loc[:,col[0]].to_numpy()
print(a)

b=df.loc[:,col[1]].to_numpy()
print(b)

c=df.loc[:,col[2]].to_numpy()
print(c)

print(df)

df.iloc[1,:]=[2,5,6,8]
print(df)
print("---------------------------------")
df.iloc[3]=20
print(df)
```
### output
```
[12 15 11 14  9 13]
[ 5  8  7  8  4 10]
[155  88 210 154 160 145]
   length  width  intensity  label
0      12      5        155      1
1      15      8         88      1
2      11      7        210      1
3      14      8        154      2
4       9      4        160      2
5      13     10        145      2
   length  width  intensity  label
0      12      5        155      1
1       2      5          6      8
2      11      7        210      1
3      14      8        154      2
4       9      4        160      2
5      13     10        145      2
---------------------------------
   length  width  intensity  label
0      12      5        155      1
1       2      5          6      8
2      11      7        210      1
3      20     20         20     20
4       9      4        160      2
5      13     10        145      2
```
### Note: set header=none if first raw doesn't related to name 

## read data with pandas

```
print("----read data with pandas----")
filename=r'/content/dataset-uci1.xlsx'
df= pd.read_excel(filename)
df
# print(df.head(10))
```
## convert to tensor
```
data= torch.from_numpy(data).type(torch.float32)

label= torch.from_numpy(label).type(torch.float32)
``` 
## read .mat data 

### SciPy

The SciPy library is a powerful, open-source Python library for scientific and technical computing. It's built on top of NumPy, which provides the foundation for working with multidimensional arrays, and it offers a vast collection of algorithms and tools for a wide range of scientific and engineering tasks.

SciPy is often used alongside other libraries like Matplotlib for plotting and Pandas for data manipulation, creating a comprehensive environment for data analysis and scientific research in Python.

for read data of .mat we use module io of this library 

```
import torch
from scipy import io
filename=r'/content/opp_hl.mat'
ds= io.loadmat(filename)
ds.keys()

```
### output

```
dict_keys(['__header__', '__version__', '__globals__', 'data_opp_hl']) ,
```
```
filename=r'/content/opp_hl.mat'
dic={'features':data,'target':label}
io.savemat(filename,dic)
#also we can save by pytorch : torch.save (dic,filename)
ds= io.loadmat(filename)
ds.keys()
data= ds['features']
label= ds['target']
data = torch.from_numpy(data).type(torch.FloatTensor)
label = torch.from_numpy(label).type(torch.LongTensor)
```
---
## Example:

reading dataset [air quality](https://www.kaggle.com/code/prakharjadaun/how-to-approach-a-dataset-eda-learn-with-me)
with pandas and then seprate data and output and put in two seprate array. draw output and put data in tensor
```
# reading the dataset

import pandas as pd
df = pd.read_csv('/content/AirQuality.csv',sep=";", decimal=",",header=0)
df.head()


# shape of our dataset
# df.shape
print("No of rows in dataset:",df.shape[0])
print("No of columns in dataset:",df.shape[1])

# getting the dtypes of the all columns
df.dtypes

# getting the numerical estimates of all the numerical column
df.describe()
df.info()

# checking null values in our dataset
df.isna().sum()
```

```
# visualizing the na values using heatmap
sns.heatmap(df.isna(),yticklabels=False,cmap='crest')
plt.show()

```
```
# getting the datatypes of all the columns having null values
# first storing all the column names in a list having null values
col = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)','PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)','PT08.S5(O3)', 'T', 'RH', 'AH']
df = df[col]
df[col].dtypes
df[col].head()
```

Since the data is real valued, we should replace all the null values with mean of each column

```
for i in col:
    df[i] = df[i].fillna(df[i].mean())
```
```
# plotting a boxplot
plt.figure(figsize=(5,5))
sns.boxplot(data=df)
plt.xticks(rotation='vertical')
plt.show()
```

## Handling Outliers

Outliers are extreme values that stand out greatly from the overall pattern of values in a dataset.
It is necessary to handle outliers because they affect the model's distribution and ultimately the model's accuracy.
We can use boxplot to get an idea of the outliers present in the dataset.
The box represents the interquartile range (IQR), with the median indicated by a line inside the box.
Any points outside the whiskers are considered potential outliers

```
# plotting a boxplot
plt.figure(figsize=(5,5))
sns.boxplot(data=df)
plt.xticks(rotation='vertical')
plt.show()
```

### To handle outliers we can follow three steps:

Trimming or removing outliers permanently.
Quantile based flooring and capping.
Replacing them with mean and median.