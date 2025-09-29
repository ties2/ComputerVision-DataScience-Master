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

## customized dataset

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
