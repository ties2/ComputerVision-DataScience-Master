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