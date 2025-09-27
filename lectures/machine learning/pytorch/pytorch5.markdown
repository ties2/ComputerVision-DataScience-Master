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