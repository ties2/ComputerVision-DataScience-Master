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
