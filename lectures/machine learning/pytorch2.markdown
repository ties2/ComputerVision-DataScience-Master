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
