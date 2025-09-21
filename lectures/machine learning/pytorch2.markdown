## PyTorch Learning Notes part 2

round numbers in a PyTorch:

You can round numbers in a PyTorch tensor using several different methods, each for a specific type of rounding
```
import torch

def demonstrate_rounding_methods():
    """
    Demonstrates different ways to round numbers in a PyTorch tensor.
    """
    # Create a tensor with a mix of positive and negative float values
    t = torch.tensor([2.3, 2.7, -1.2, -1.8, 4.5, -3.5])
    print(f"Original Tensor:\n{t}\n")

    # Round to the nearest integer
    rounded_t = torch.round(t)
    print(f"torch.round(t): rounds to nearest integer")
    print(f"Result: {rounded_t}\n")

    # Round up to the nearest integer
    ceil_t = torch.ceil(t)
    print(f"torch.ceil(t): rounds up")
    print(f"Result: {ceil_t}\n")

    # Round down to the nearest integer
    floor_t = torch.floor(t)
    print(f"torch.floor(t): rounds down")
    print(f"Result: {floor_t}\n")

    # Truncate the decimal part (round towards zero)
    trunc_t = torch.trunc(t)
    print(f"torch.trunc(t): truncates (rounds towards zero)")
    print(f"Result: {trunc_t}\n")

if __name__ == "__main__":
    demonstrate_rounding_methods()
```

# Original Tensor:
tensor([ 2.3000,  2.7000, -1.2000, -1.8000,  4.5000, -3.5000])

torch.round(t): rounds to nearest integer
Result: tensor([ 2.,  3., -1., -2.,  4., -4.])

torch.ceil(t): rounds up
Result: tensor([ 3.,  3., -1., -1.,  5., -3.])

torch.floor(t): rounds down
Result: tensor([ 2.,  2., -2., -2.,  4., -4.])

torch.trunc(t): truncates (rounds towards zero)
Result: tensor([ 2.,  2., -1., -1.,  4., -3.])

---
### squeeze

In PyTorch, torch.squeeze() is a tensor operation that simplifies the shape of a tensor by removing dimensions that have a size of 1.  This is a crucial operation, especially when dealing with data that has been prepared for machine learning models, as it helps remove unnecessary dimensions that don't hold any meaningful data.

Why is squeeze() useful?

Simplifying Tensor Shapes: It helps reduce the number of dimensions, making the tensor easier to work with. For example, a tensor with shape (1, 28, 28) might represent a single-channel image. Squeezing this would result in a tensor of shape (28, 28), which is much more natural for many operations.

Compatibility: Many functions and models expect tensors with a specific rank (number of dimensions). squeeze() allows you to match these requirements easily. For example, a loss function might expect an output with shape (batch_size, num_classes), but your model might output (batch_size, 1, num_classes). You would use squeeze() to remove the extra dimension.

squeeze() vs. unsqueeze()

squeeze() removes dimensions of size 1.

unsqueeze() adds a dimension of size 1.

```
import torch

# Create a tensor with multiple dimensions of size 1
x = torch.zeros(2, 1, 3, 1, 4)
print(f"Original tensor shape: {x.shape}")

# Use squeeze() without specifying a dimension
y = torch.squeeze(x)
print(f"Shape after squeezing all dimensions of size 1: {y.shape}")

# Create a new tensor to demonstrate squeezing a specific dimension
z = torch.zeros(1, 5, 1, 2)
print(f"\nOriginal tensor shape: {z.shape}")

# Squeeze only the first dimension (index 0)
# Note: The dimension at index 2 will remain because it was not specified.
w = torch.squeeze(z, dim=0)
print(f"Shape after squeezing only dimension 0: {w.shape}")

# Squeeze only the third dimension (index 2)
v = torch.squeeze(z, dim=2)
print(f"Shape after squeezing only dimension 2: {v.shape}")
```

### output:
Original tensor shape: torch.Size([2, 1, 3, 1, 4])
Shape after squeezing all dimensions of size 1: torch.Size([2, 3, 4])

Original tensor shape: torch.Size([1, 5, 1, 2])
Shape after squeezing only dimension 0: torch.Size([5, 1, 2])
Shape after squeezing only dimension 2: torch.Size([1, 5, 2])

---
## transpose

torch.transpose() is a PyTorch function used to swap two dimensions of a tensor. Unlike a simple matrix transpose where rows become columns and vice-versa, torch.transpose() is more general and works for tensors of any number of dimensions.

```
import torch

def demonstrate_transpose(tensor, dim0, dim1):
    """
    Demonstrates the transpose operation on a given tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        dim0 (int): The first dimension to swap.
        dim1 (int): The second dimension to swap.
    """
    print(f"Original Tensor:\n{tensor}")
    print(f"Original Shape: {tensor.shape}\n")
    
    # Perform the transpose operation
    transposed_tensor = torch.transpose(tensor, dim0, dim1)
    
    print(f"Transposed Tensor (swapping dim {dim0} and dim {dim1}):\n{transposed_tensor}")
    print(f"New Shape: {transposed_tensor.shape}\n")
    
    # Note on underlying storage: transpose() returns a new tensor that shares
    # the same underlying data storage with the original tensor. This means
    # that an in-place modification to one will affect the other.
    print("-" * 30)

# --- Example 1: Transposing a 2D Tensor (Matrix) ---
print("--- Example 1: 2D Tensor Transpose (like a matrix) ---")
# A 2x3 tensor
tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])

# Swapping dimension 0 (rows) and dimension 1 (columns)
demonstrate_transpose(tensor_2d, 0, 1)

# --- Example 2: Transposing a 3D Tensor ---
print("--- Example 2: 3D Tensor Transpose ---")
# A 2x3x4 tensor
tensor_3d = torch.arange(24).reshape(2, 3, 4)

# Swapping dimension 0 (the 'batch' or 'z-axis') and dimension 2 (the 'columns' or 'x-axis')
# The shape will change from (2, 3, 4) to (4, 3, 2)
demonstrate_transpose(tensor_3d, 0, 2)


# --- Example 3: Comparison with .permute() ---
print("--- Example 3: Comparison with .permute() ---")
# .permute() is a more general version of transpose()
# It allows you to rearrange all dimensions in any order.

tensor_3d_permute = torch.arange(24).reshape(2, 3, 4)
print(f"Original Tensor for permute:\n{tensor_3d_permute}")
print(f"Original Shape: {tensor_3d_permute.shape}\n")

# Same operation as Example 2, but using .permute()
# The original dimensions were at indices [0, 1, 2].
# We want to swap dim 0 and dim 2, so the new order is [2, 1, 0].
permuted_tensor = tensor_3d_permute.permute(2, 1, 0)
print(f"Permuted Tensor (new order [2, 1, 0]):\n{permuted_tensor}")
print(f"New Shape: {permuted_tensor.shape}")
```

## reshape

torch.reshape() is a PyTorch function that returns a new tensor with the same data as the input tensor but with a different shape. The key difference between reshape() and operations like transpose() or permute() is that reshape() keeps the order of the elements the same, simply reinterpreting them into a new shape. This operation is most efficient when the underlying data is stored contiguously in memory.

use reshape() when you need to change the shape of a tensor without altering the sequence of its elements in memory. Use transpose() or permute() when you need to swap or reorder the dimensions. You might need to use .contiguous() before reshape() if a previous operation like transpose() made the tensor's memory layout non-contiguous.

```
# This script demonstrates the use of torch.reshape()
# and the concept of memory contiguity in PyTorch.

import torch

def demonstrate_reshape(tensor):
    """
    Shows how reshape() works on a tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor.
    """
    print(f"Original Tensor:\n{tensor}")
    print(f"Original Shape: {tensor.shape}")
    print(f"Is Contiguous: {tensor.is_contiguous()}\n")
    
    # Reshape the tensor into a new shape (e.g., from 2x3 to 3x2)
    try:
        reshaped_tensor = tensor.reshape(3, 2)
        print(f"Reshaped Tensor:\n{reshaped_tensor}")
        print(f"New Shape: {reshaped_tensor.shape}")
        print(f"Is Contiguous: {reshaped_tensor.is_contiguous()}\n")
    except RuntimeError as e:
        print(f"Could not reshape due to a RuntimeError:\n{e}")
    
    print("-" * 30)

# --- Example 1: Reshaping a simple, contiguous tensor ---
print("--- Example 1: Reshaping a Contiguous Tensor ---")
# A simple 2x3 tensor is contiguous by default
contiguous_tensor = torch.arange(6).reshape(2, 3)
demonstrate_reshape(contiguous_tensor)

# --- Example 2: The effect of transpose() on contiguity ---
print("--- Example 2: Reshaping a Transposed (Non-Contiguous) Tensor ---")
# First, create a tensor and transpose it.
# This swaps the dimensions but does NOT make the data contiguous in memory.
transposed_tensor = contiguous_tensor.transpose(0, 1)

# Note that the order of elements has been changed.
print(f"Tensor after transpose(0, 1):\n{transposed_tensor}")
print(f"Shape: {transposed_tensor.shape}")
print(f"Is Contiguous: {transposed_tensor.is_contiguous()}\n")

# Now, try to reshape this non-contiguous tensor.
# PyTorch will automatically create a contiguous copy in the background.
# It's an important distinction for performance.
print("Attempting to reshape the non-contiguous tensor...")
reshaped_from_transposed = transposed_tensor.reshape(2, 3)
print(f"Reshaped tensor from transposed one:\n{reshaped_from_transposed}")
print(f"New Shape: {reshaped_from_transposed.shape}")
print(f"Is Contiguous: {reshaped_from_transposed.is_contiguous()}\n")
print("-" * 30)

# --- Example 3: Making a tensor contiguous explicitly ---
print("--- Example 3: Using .contiguous() before reshape() ---")
# For clarity and performance, it's often best to make a tensor contiguous yourself.
# This forces the memory to be rearranged before the reshape operation.
tensor_to_reshape = contiguous_tensor.transpose(0, 1).contiguous()

print(f"Original tensor (transposed then contiguous):\n{tensor_to_reshape}")
print(f"Shape: {tensor_to_reshape.shape}")
print(f"Is Contiguous: {tensor_to_reshape.is_contiguous()}\n")

# Now reshape() is guaranteed to be a fast, zero-copy operation.
explicit_reshaped = tensor_to_reshape.reshape(2, 3)
print(f"Tensor after .contiguous().reshape(2, 3):\n{explicit_reshaped}")
print(f"New Shape: {explicit_reshaped.shape}")
print(f"Is Contiguous: {explicit_reshaped.is_contiguous()}\n")

# --- Special case: using -1 ---
print("--- Special Case: Using -1 in reshape() ---")
# PyTorch can infer one dimension's size if you pass -1.
# This is very useful when you want to flatten a tensor.
tensor_for_flatten = torch.arange(12).reshape(2, 2, 3)
print(f"Original 3D tensor:\n{tensor_for_flatten}")
print(f"Original Shape: {tensor_for_flatten.shape}")
flattened_tensor = tensor_for_flatten.reshape(-1) # Flattens to a 1D tensor
print(f"Flattened tensor using reshape(-1):\n{flattened_tensor}")
print(f"New Shape: {flattened_tensor.shape}")
```

