# PyTorch Learning Notes part 5
## work with module glob for reading data

Key Keywords
### File Handling

* glob: Find files using path patterns.

* glob.glob(pathname + '/*'): Get all files in a directory using a * (wildcard).

### Python OOP (Dunder Methods)

* class: A blueprint for creating objects.

* Dunder Methods: Special methods triggered by Python operations.

  * __init__(self, ...): The constructor; runs when an object is created to set initial state (e.g., self.data = ...).

  * __len__(self): Returns the size; triggered by len(object).

  * __getitem__(self, index): Gets an item; triggered by object[index].

  * __call__(self, ...): Makes the object callable; triggered by object(...).

### PyTorch Data

* Dataset (Abstract Class): The "data source"; holds all data.

  * Must implement __len__ (total size) and __getitem__ (get one sample).

  *  datasets.MNIST: A built-in example.

  * TensorDataset: A simple wrapper for existing tensors.

* DataLoader (Utility Class): The "data delivery"; wraps a Dataset.

  * Provides: Batching, Shuffling, and Multiprocessing.

  * Benefit: GPU efficiency and stable training.

### Data Preprocessing (Transforms)

* transforms.Compose([...]): A pipeline that chains multiple transforms together.

* transforms.ToTensor(): Converts data (like NumPy arrays) into a PyTorch tensor.

* transforms.Normalize(): Rescales tensor data (e.g., to mean 0, std 1).

* Lambda: Used for applying custom functions (e.g., one-hot encoding labels).

* Label Transform: Crucial for converting labels to torch.long, which is required by loss functions like CrossEntropyLoss.

### Batching & DataLoader Params

* Epoch: One complete pass through the entire dataset.

* Batch Size: The number of samples processed at one time.

* shuffle=True: Randomizes the data order at the start of every epoch.

* drop_last=True: Discards the final, smaller batch if the dataset size isn't perfectly divisible by the batch size.

* num_workers > 0: Uses parallel processing to load data in the background, preventing a CPU bottleneck and keeping the GPU busy.


## Example without glob

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

when we want to read files with different names it can be difficult, in this situation we use glob library
```
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

pathname='/content'
filename= glob.glob(pathname+'/*')
print(filename)
```
### output:
['/content/S040.txt',..., '/content/sample_data']

or for subfolder:
pathname='/content/*/*'
filename= glob.glob(pathname+'/*')
print(filename)

### output:
['/content/s/S040.txt',..., '/content/sample_data']

---
# Class and object oriented

## What are "Dunder Methods"?
The methods surrounded by double underscores (e.g., __init__, __len__) are known as Dunder Methods (short for Double Underscore methods) or Magic Methods.

You almost never call these methods directly (you wouldn't write my_object.__len__()). Instead, they are automatically invoked by Python when you use a specific built-in function or operator.

By defining them, you allow your custom objects to implement a "protocol," enabling them to behave like built-in types (like lists, dictionaries, or numbers).

__init__(self, ...): The Constructor
This is the most common dunder method and is essential for almost any class you write.

Role: The constructor. It's the first method that runs when you create a new instance of a class.

Purpose: To set the initial state of the object by accepting arguments and assigning them to instance attributes (like setting a name or an age).

Triggered by: Instantiating the class: ClassName(...)

Example:

* note: method __init__  define in class and during make object it call automatically and it's code run

```
class Car:
    # This method runs automatically when we create a Car object.
    def __init__(self, make, model):
        self.make = make      # Initializing the 'make' attribute
        self.model = model    # Initializing the 'model' attribute
        self.running = False

# This line calls the __init__ method behind the scenes
my_car = Car("Tesla", "Model 3")
print(my_car.make)
# Output: Tesla
```

__len__(self): Sizing an Object
This method allows your object to have a measurable size, adhering to the "Sizing Protocol."

Role: Must return an integer representing the length or size of the object.

Purpose: Allows you to use the built-in len() function on your custom objects.

Triggered by: len(my_object)

Example:
```
class Bookcase:
    def __init__(self, book_list):
        self.books = book_list

    # This method runs automatically when we call len() on a Bookcase object.
    def __len__(self):
        return len(self.books)

shelf = Bookcase(["The Martian", "Dune", "1984"])
print(len(shelf))
# Output: 3
```
### __len__(self): Sizing an Object
This method allows your object to have a measurable size, adhering to the "Sizing Protocol."

Role: Must return an integer representing the length or size of the object.

Purpose: Allows you to use the built-in len() function on your custom objects.

Triggered by: len(my_object)

Example:
```
class Bookcase:
    def __init__(self, book_list):
        self.books = book_list

    # This method runs automatically when we call len() on a Bookcase object.
    def __len__(self):
        return len(self.books)

shelf = Bookcase(["The Martian", "Dune", "1984"])
print(len(shelf))
# Output: 3
```
### __getitem__

Example:
```
class SimpleContainer:
    """A container that allows access by index."""
    def __init__(self, data_list):
        self.data = data_list

    # This method runs automatically when square brackets are used.
    def __getitem__(self, index):
        # The key is the index used inside the square brackets.
        return self.data[index]

my_data = SimpleContainer(['Apple', 'Banana', 'Cherry', 'Date'])

# This line calls my_data.__getitem__(2)
print(my_data[2])

# Output: Cherry
```

Other Essential Dunder Methods
Many other magic methods allow you to implement different Python protocols:

| Dunder Method        | Protocol /Role                  | Triggered By | Example usage
| -------------| ------------- | ------------- |---------------------------- |
| __str__(self)	| String Representation (Readable) | print(my_object) or str(my_object)	| Returns a human-readable string for printing.|
| __repr__(self)| String Representation (Official)	| Displaying object in a console/debugger.|	Returns an unambiguous string, often useful for recreating the object.
| __getitem__(self, key)|	Container Access	|my_object[key]	| Allows item retrieval by index (like a list) or key (like a dict).|
| __setitem__(self, key, value)	| Container Assignment	| my_object[key] = value |	Allows assignment to an index or key.
| __add__(self, other) |	Numeric Addition	| obj1 + obj2	| Defines how the + operator behaves for your object.
| __eq__(self, other) |	Comparison |	obj1 == obj2 |	Defines what it means for two custom objects to be equal. |

----
## dataset and dataloaders 
This is a core concept in PyTorch for efficiently managing and feeding data into your machine learning models. They work in tandem: the Dataset handles what the data is, and the DataLoader handles how it's delivered.

1. Dataset (The Data Source)
The Dataset is the class responsible for holding your data samples and their corresponding labels. Think of it as the storage locker for your entire dataset.

Every custom PyTorch Dataset you create must implement two special (dunder) methods:

__len__: Returns the total number of samples in the dataset.

__getitem__: Returns a single data sample and its label when given an index (e.g., my_dataset[5]).

The Dataset ensures that you can access any item in your data by index.

2. DataLoader (The Delivery Utility)
The DataLoader is an iterable wrapper around your Dataset. It transforms the static dataset into an efficient, dynamic stream of data ready for model training. Think of it as the assembly line that prepares batches of data.

It provides critical functionality needed for training:

Batching: It groups individual samples from the Dataset into manageable mini-batches (e.g., 32 or 64 samples at a time) for efficient processing by the GPU.

Shuffling: It randomizes the order of the data between training epochs to prevent the model from learning the sequence of the data.

Multiprocessing: It uses multiple worker threads to load data in the background, preventing your GPU from sitting idle while waiting for the next batch (a common bottleneck).

The Partnership
You instantiate the Dataset once to structure all your data, and then you wrap that Dataset in a DataLoader to handle the batching and feeding process during training

benefits of dataloader:
* let user load data parallel
* data augmentation
* flexibility
* shuffling

 

```
from torch.utils.data import Dataset,DataLoader

class CustomDataset(Dataset):
  def __init__(self,<arguments>):
   pass
  def __len__(self):
   pass
  def __getitem__(self,index):
   pass
```

* dataset is an abstract class which responsile for load data from resource and doing preporesseing and change to tensor.

Note:

* Abstract Class
Role: Defines a contract or blueprint. It specifies what methods derived classes must implement, but it doesn't provide a complete implementation itself.

Key Feature: You cannot create an instance (object) of an abstract class. It exists only to be inherited from.

Use Case: Establishing a common interface for a group of related classes (e.g., a base Animal class defining an abstract make_sound() method).

* Utility Class
Role: Holds a collection of static methods (or functions) that perform common, reusable operations, usually on data provided as arguments.

Key Feature: It has no internal state (no self.attribute). You typically never create an instance of a utility class; you just call its methods directly using the class name.

Use Case: Grouping helpful, generic tools (e.g., a MathUtils class with static methods like calculate_distance or format_currency).

---
## pytorch dataset

# try MNIST dataset as a sample
it is image dataset include handwriting sample of digits.
```
import torch
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import seaborn as sns

train_ds= datasets.MNIST(root='content/data',train=True,download=True,transform=ToTensor(),target_transform=None)
test_ds= datasets.MNIST(root='content/data',train=False,download=True,transform=ToTensor(),target_transform=None)
len(test_ds)

for i in range(10):

  sample,label=train_ds[i]

  plt.subplot(2,5,i+1)

  plt.imshow(sample.squeeze(), cmap='gray')

  plt.title(label,fontsize=15)

plt.tight_layout()

plt.show()

```
### output

<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/images/MNIST%20sample%20digit.png" alt="Computer Vision Logo" width="300" />
</p>

## customized dataset,dataloaders

Example:

```
#dataset EEN ( has two file s and z and each of them has some *.txt files)

import torch
from torch.utils.data import Dataset,DataLoader
import glob
import pandas as pd

class CustomDataset(Dataset):
  def __init__(self,pathname,transforms=None,target_transform=None):
    self.file_names= glob.glob(pathname+'/*/*.txt') # Modified glob pattern
    self.class_map= {'S':0,'Z':1} # Changed keys to uppercase
    self.transforms=transforms
    self.target_transform=target_transform

  def __len__(self):
    return len(self.file_names)

  def __getitem__(self,index):
    filename= self.file_names[index]
    df= pd.read_csv(filename,header=None)
    chr=filename.split('/')[-1][0] # Modified split character

    sample=df.iloc[:,0].to_numpy()
    label=self.class_map[chr]
    if self.transforms:
      sample=self.transforms(sample)
    if self.target_transform:
      label=self.target_transform(label)
    return sample,label

pathname='content'
ds=CustomDataset(pathname,transforms=None, target_transform=None)
len(ds)
sampl,label= ds[1]
print(sampl,label)
```
Example:
```
import torch
from torch.utils.data import Dataset,DataLoader
import glob
import pandas as pd

class CustomDataset(Dataset):
  def __init__(self,pathname,transforms=None,target_transform=None):
    df= pd.read_excel(pathname)
    data=df.iloc[:,:-1].to_numpy()
    label=df.iloc[:,-1].to_numpy()
    self.data=data
    self.target=label
    self.transforms=transforms
    self.target_transform=target_transform

   

  def __len__(self):
    return len(self.target)

  def __getitem__(self,index):
    sample= self.data[index]
    label=self.target[index]
    if self.transforms:
      sample=self.transforms(sample)
    if self.target_transform:
      label=self.target_transform(label)
    return sample,label
    

pathname='content/iris.xlsx'
ds=CustomDataset(pathname,transforms=None, target_transform=None)
len(ds)
sampl,label= ds[11]
print(sampl,label)
```
In machine learning, the expression ds = list(zip(data, target)) is commonly used in Python to create a dataset by pairing input data (data) with corresponding labels or outputs (target). Let's break it down:

data: This typically represents the feature matrix, often a 2D array-like structure (e.g., a NumPy array or list of lists) where each row is a sample and each column is a feature.
target: This is the array or list containing the labels or output values corresponding to each sample in data.
zip(data, target): The zip function pairs each sample in data with its corresponding label in target, creating an iterator of tuples where each tuple contains a feature vector and its label.
list(): Converts the zip iterator into a list of tuples, where each tuple is of the form (feature_vector, label).
ds: The resulting list is assigned to the variable ds, which can be thought of as a dataset where each element is a pair of input features and their corresponding target value.

```
data = [[1, 2], [3, 4], [5, 6]]  # Feature matrix (3 samples, 2 features each)
target = [0, 1, 0]               # Corresponding labels
ds = list(zip(data, target))
```
The result of ds would be:
```
[([1, 2], 0), ([3, 4], 1), ([5, 6], 0)]

```
Common Use Cases

Data Preparation: This format is useful for iterating over data and labels together, especially in custom loops for training machine learning models or for shuffling datasets.
Shuffling: When paired with random.shuffle(ds), it ensures that features and labels remain aligned while randomizing the order of samples.
Custom Datasets: Libraries like PyTorch or TensorFlow often expect datasets in a format where inputs and outputs are paired, and list(zip(data, target)) is a simple way to achieve this.

second method:
```
from torch.utils.data import TensorDataset
data = [[1, 3], [2, 6], [1, 2]]  # Feature matrix (3 samples, 2 features each)
target = [0, 1, 0]               # Corresponding labels
data = torch.tensor(data, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.long)
ds1=TensorDataset(data,target)
sampl,target= ds1[2]
print(sampl,target)

```
output:

tensor([5., 6.]) tensor(0)

---

## call
The __call__ method in Python is a special dunder method that allows an instance of a class to be treated like a function.

It makes your objects "callable."

__call__(self, ...)
Role: Makes an object instance callable.

Triggered By: Using parentheses () on an object instance.

Purpose:

Simplification: Allows complex objects to have a primary action, making the API cleaner (e.g., instead of optimizer.step(), you might just call optimizer()).

State Management: It allows you to run a function that remembers and uses the state (attributes) stored inside the object instance.

PyTorch Modules: In PyTorch, every custom neural network model must inherit from nn.Module. When you run output = model(input_data), you are actually calling the model's inherited __call__ method, which in turn executes the forward() method you defined.

Simple Example:
```
class Multiplier:
    def __init__(self, factor):
        self.factor = factor # State saved in the object

    # This runs when you call the object instance like a function
    def __call__(self, x):
        return x * self.factor

# 1. Create the callable object (state is set to 5)
double = Multiplier(2)
triple = Multiplier(3)

# 2. Call the object (This executes the __call__ method)
print(double(10)) # Output: 20
print(triple(10)) # Output: 30
```
Example to define class totensor and use call method

```
import torch
import numpy as np

class ToTensor:
    """
    A callable class that converts input data (list or NumPy array)
    into a PyTorch float tensor using the __call__ method.
    """
    def __init__(self):
        # The constructor does nothing here, as the state is simple (no attributes needed).
        pass

    # This method is automatically executed when the object instance is called like a function.
    def __call__(self, data):
        """Converts the input data into a PyTorch tensor."""
        print(f"Input Type received: {type(data)}")
        
        # Convert to tensor and ensure it's a float type (common for model input)
        tensor_output = torch.tensor(data, dtype=torch.float32)
        
        print(f"Output Tensor Shape: {tensor_output.shape}")
        return tensor_output

# --- Demonstration ---

# 1. Create an instance of the class
data_converter = ToTensor()

# 2. Prepare sample data (NumPy array)
sample_data = np.array([[1.0, 2.0], [3.0, 4.0]])

print("\n--- Calling the object like a function ---")

# 3. Call the instance directly to execute the __call__ method
tensor_result = data_converter(sample_data)

print("\nResulting Tensor:")
print(tensor_result)

```
Output:


--- Calling the object like a function ---
Input Type received: <class 'numpy.ndarray'>
Output Tensor Shape: torch.Size([2, 2])

Resulting Tensor:
tensor([[1., 2.],
        [3., 4.]])

---
## Normalization

Normalization can be effective in handling data that contains outliers.
On the other hand, normalization can increase the speed of convergence in neural networks.
However, it is better to evaluate your model once without normalization and once with normalization and choose the mode that performs best.

Normalization is a data preprocessing technique that adjusts the scale of numerical features in a dataset to fit within a common, defined range.

1. What is Normalization?
It's the process of rescaling data so that the value of every feature lies between 0 and 1.

 
2. Why is it Necessary?
The main goal is to prevent features with naturally large numerical ranges from dominating the learning process.

Imagine a model trying to learn from two features:

Annual Income: (e.g., 20,000 to 200,000)

Age: (e.g., 20 to 60)

Without normalization, the Income feature's massive numerical range would have a much greater influence on the model's loss function and gradient calculations, making the model insensitive to changes in Age. Normalization ensures all features contribute proportionally.

3. Normalization vs. Standardization
While often used interchangeably, there's a subtle difference:

Normalization (Min-Max Scaling): Rescales data to be strictly between a minimum and maximum value, typically [0,1].

Standardization (Z-Score): Rescales data so it has a mean of 0 and a standard deviation of 1. This is generally preferred for algorithms that assume a normal distribution.

## Compose
1. What is Compose?

It's a class from the torchvision.transforms module that acts like a pipeline. It takes a list of individual transformation objects (like ToTensor, Normalize, Resize, etc.) and runs the data through them in the exact order they are defined.

2. Why use it?
Automation: It automates the entire preparation sequence for every single data sample requested by the DataLoader.

Consistency: It ensures that every image, text snippet, or row of data goes through the exact same set of steps (e.g., resizing, then cropping, then converting to tensor, then normalizing).

Clean Code: It keeps your Dataset class clean by moving the complex, multi-step preprocessing logic out of the core data retrieval (__getitem__) method.

Example:
```
import torch
import torchvision.transforms as T 
import torchvision.datasets as datasets 

transform1 = T.ToTensor()

transform2 = T.Normalize(
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5]
)

transform = T.Compose([transform1, transform2])

print(f"Sequence: {transform}")
```
# Make transfor for label

The reason we specifically transform labels (targets) in PyTorch is not necessarily to change their value, but to ensure they have the exact data type and structure that PyTorch's loss functions and training mechanisms require.

Why Transform the Label?
The transformation exists to satisfy the strict requirements of PyTorch's core mathematical operations, especially for classification tasks.

1. The Requirement: torch.long

The most crucial reason is that standard classification loss functions in PyTorch, like nn.CrossEntropyLoss, require the target labels (y) to be an integer type, specifically a 64-bit integer, which corresponds to the data type torch.long.

Why? In classification, the labels represent indices (e.g., 0, 1, 2,...) pointing to the predicted class scores (logits) in the model's output. Loss functions use these integer indices to look up the correct prediction score and calculate the error.

2. Consistency

Data read from files, NumPy arrays, or Pandas DataFrames often defaults to types like standard Python integers, float64, or int32.

The label transformation process, demonstrated by the LabelToTensor class in the Canvas, guarantees that every label is consistently converted to a PyTorch tensor of the precise torch.long type before it's used in the training loop. This prevents runtime errors that would otherwise occur when the model tries to compute the loss.

```
import torch
import numpy as np

class LabelToTensor:
    # This is the method executed when the object is called like a function.
    def __call__(self, label):
        return torch.tensor(label, dtype=torch.long)

# --- Demonstration ---

# 1. Create the label transformation object
target_transform = LabelToTensor()

# 2. Sample raw label data
raw_label_int = 5
raw_label_np = np.array([1, 0, 3])

# 3. Apply the transform by calling the object directly
tensor_label_1 = target_transform(raw_label_int)
tensor_label_2 = target_transform(raw_label_np)

print(f"Raw Label (Int): {raw_label_int} -> Tensor Type: {tensor_label_1.dtype}, Value: {tensor_label_1}")
print(f"Raw Label (NumPy): {raw_label_np} -> Tensor Type: {tensor_label_2.dtype}, Value: {tensor_label_2}")
```
# Example for using MNIST by check lambda,totensor and normalize

```
import torch
import torchvision.transforms as T
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor,Lambda,Normalize,Compose
import matplotlib.pyplot as plt
import seaborn as sns


train_ds= datasets.MNIST(root='content/data',train=True,download=True,transform=Compose(ToTensor(),Normalize(0,1)),target_transform=lambda y:torch.zeros(10,dtype=torch.float).scatter_(0,torch.tensor(y),value=1).type(torch.float32))
test_ds= datasets.MNIST(root='content/data',train=False,download=True,transform=ToTensor(),target_transform=lambda y:torch.zeros(10,dtype=torch.float).scatter_(0,torch.tensor(y),value=1).type(torch.float32))
len(test_ds)


sample,label=train_ds[1]
plt.imshow(sample.squeeze(), cmap='gray')
plt.show()
```
# dataset batching 

Why Batching is Essential
Instead of feeding one data sample at a time to the model (which is slow and inefficient), batching achieves two main goals:

* Efficiency (GPU): Modern hardware, especially GPUs, is highly optimized for parallel processing. Feeding a large chunk of data (a batch) simultaneously is vastly faster than processing samples one by one.

Stable Training (Gradient Descent): When the model calculates the error (loss) and adjusts its weights (gradient descent), it needs this calculation to be stable.

Using a single data point gives a noisy, unstable estimate of the true error.

Using a mini-batch (e.g., 32 or 64 samples) averages out the noise across those samples, giving a much more stable and accurate estimate of the direction the model should adjust its weights.

In PyTorch, the DataLoader handles all the batching logistics, automatically grabbing individual samples from the Dataset and compiling them into these mini-batches.

|Term	Short | Explanation |
| ---------- | -------------- |
| Training	| The overall process where the model is shown the data, makes predictions, calculates the error (loss), and adjusts its internal weights to minimize that error.
| Epoch	| One complete pass through the entire training dataset. If your dataset has 10,000 samples and you train for 10 epochs, the model sees 100,000 total samples.
| Training Loop	| The repetitive, sequential code block that defines the steps for one epoch: 1. Load Batch → 2. Forward Pass (predict) → 3. Calculate Loss → 4. Backpropagation (adjust weights).
| Batch Size	| The number of data samples processed together at one time. If your Batch Size is 32, the model adjusts its weights after seeing 32 samples. (Chosen by the user).
| Batch Number | The sequential count of mini-batches processed within a single epoch. If you have 1,000 total samples and a batch size of 100, you will have 10 batches (and 10 batch numbers) per epoch.


Batching is a mechanical process handled by the DataLoader to prepare data for efficient training. It involves three simple, sequential steps:

1. Request Indices: The DataLoader first generates a list of random indices from the total dataset size (e.g., indices [45, 12, 98, ...] for a batch size of 32). This ensures shuffling.

2. Retrieve Samples: It uses the __getitem__ method of the Dataset to fetch the individual data sample (X) and its label (y) for each of those indices.

3. Collate & Stack: It then takes these individual samples (e.g., 32 images) and stacks them together to form two larger tensors:

* One tensor for all features (Batch X).

* One tensor for all labels (Batch y).

This resulting mini-batch is then immediately passed to the model for the forward pass. This process repeats until the entire dataset (one epoch) has been consumed.


### PyTorch DataLoader that control how batches are prepared:

### shuffle

1. shuffle=True (Randomization)
* What it does: Randomly shuffles the indices of the entire dataset before each new training epoch.

* Purpose: Prevents the model from learning the sequence or order of the data points. If the data were always in the same order, the model might optimize itself specifically for that sequence, leading to poor generalization on new, unseen data.

* Result: A different sequence of mini-batches is generated for every epoch.

### drop_last

2. drop_last=True (Handling Leftovers)
* What it does: Tells the DataLoader to discard the very last batch of the dataset if its size is smaller than the specified batch_size.

* Purpose: Ensures that every single batch during training has the exact same dimensions. This is crucial when:

Using distributed training (multiple GPUs).

Using stateful recurrent models (where a consistent batch size is required).

You want the batch size to be perfectly divisible into the total number of samples for easier tracking.

* Result: If your dataset has 103 samples and your batch size is 32, you would normally get three full batches (32, 32, 32) and one small batch (7). If drop_last=True, the small batch of 7 is simply discarded.


### num_workers
 
1. What it Does
It enables multiprocessing for data loading.

The workers run in the background, performing slow tasks like reading files, applying transformations (ToTensor, Normalize), and collating batches.

2. Why it's Important (The Bottleneck)
Prevents GPU Starvation: The primary purpose is to ensure the GPU never sits idle, waiting for the CPU to finish preparing the next batch of data. If num_workers is 0 (the default), the CPU loads the data and the GPU waits.

Speed: By setting num_workers to a value greater than 0, the CPU and GPU can work in parallel: the GPU trains on the current batch while the worker processes prepare the next batch in the background.

3. How to Set It
num_workers=0: Data loading is done in the main process (slowest, only used for simple debugging).

Recommended Value: A good starting point is often the number of CPU cores you have available, or perhaps half that amount. You should increase this value until your GPU utilization is consistently high. If you set it too high, you can overload your CPU and run out of memory.