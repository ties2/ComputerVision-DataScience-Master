```python
content = """# Phase 1 - Python Foundations (Days 1-14)

The goal of this phase is to master Python at a level expected from a senior engineer. Interviewers will test not just whether you can code, but whether you understand how Python works under the hood.

## 1.1 Object-Oriented Programming (OOP)
Object-oriented programming is the backbone of professional Python. You must understand classes, inheritance, encapsulation, polymorphism, and abstraction at a deep level.

### 1.1.1 Classes and Instances
A class is a blueprint for creating objects. An instance is a concrete object created from a class. Every instance has its own namespace (stored in `__dict__`) and shares the class namespace for methods.

```python
class ImageProcessor:
    supported_formats = ["jpg", "png", "bmp"]  # class attribute
 
    def __init__(self, path, target_size=(224, 224)):
        self.path = path              # instance attribute
        self.target_size = target_size
        self._cache = None            # private by convention
 
    def load(self):
        \"\"\"Load and preprocess image.\"\"\"
        import cv2
        img = cv2.imread(self.path)
        if img is None:
            raise FileNotFoundError(f"Cannot load {self.path}")
        self._cache = cv2.resize(img, self.target_size)
        return self._cache
 
    @classmethod
    def from_url(cls, url, target_size=(224, 224)):
        \"\"\"Alternative constructor from URL.\"\"\"
        import urllib.request, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        urllib.request.urlretrieve(url, tmp.name)
        return cls(tmp.name, target_size)
 
    @staticmethod
    def is_valid_format(filename):
        return filename.split(".")[-1].lower() in ImageProcessor.supported_formats

```

**Key interview points:** `@classmethod` receives the class as the first argument (`cls`) and is used for alternative constructors. `@staticmethod` receives no implicit first argument and is used for utility functions related to the class. Instance attributes live in `self.__dict__`, class attributes live in `ClassName.__dict__`.

### 1.1.2 Inheritance and MRO

Python supports multiple inheritance. The Method Resolution Order (MRO) determines which method is called when a method exists in multiple parent classes. Python uses the C3 Linearization algorithm.

```python
class BaseModel:
    def predict(self, x):
        raise NotImplementedError("Subclasses must implement predict()")
 
    def preprocess(self, x):
        return x / 255.0  # normalize to [0, 1]
 
class ClassificationMixin:
    def get_top_k(self, probs, k=5):
        import numpy as np
        indices = np.argsort(probs)[-k:][::-1]
        return [(i, probs[i]) for i in indices]
 
class ResNetClassifier(ClassificationMixin, BaseModel):
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
 
    def predict(self, x):
        preprocessed = self.preprocess(x)
        probs = self.model(preprocessed)
        return self.get_top_k(probs)
 
    def _load_model(self, path):
        # load ONNX or TorchScript model
        return None  # placeholder
 
# Check MRO:
# print(ResNetClassifier.__mro__)
# => ResNetClassifier -> ClassificationMixin -> BaseModel -> object

```

The MRO follows left-to-right, depth-first order with C3 linearization. In the example above, Python looks in `ResNetClassifier` first, then `ClassificationMixin`, then `BaseModel`, then `object`. This is critical for understanding which method gets called in diamond inheritance patterns.

### 1.1.3 Encapsulation and Properties

Python uses name mangling for private attributes (double underscore prefix) and the `property` decorator for controlled access to attributes.

```python
class ModelConfig:
    def __init__(self, learning_rate=0.001, batch_size=32):
        self._learning_rate = learning_rate
        self.__secret_key = "internal"  # name-mangled to _ModelConfig__secret_key
        self.batch_size = batch_size
 
    @property
    def learning_rate(self):
        \"\"\"Getter with validation.\"\"\"
        return self._learning_rate
 
    @learning_rate.setter
    def learning_rate(self, value):
        if not 0 < value < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {value}")
        self._learning_rate = value
 
# Usage:
# config = ModelConfig()
# config.learning_rate = 0.01   -> calls setter
# config.learning_rate          -> calls getter, returns 0.01
# config.learning_rate = 5      -> raises ValueError

```

### 1.1.4 Abstract Base Classes

Use `abc.ABC` and `@abstractmethod` to define interfaces that subclasses must implement. This is essential for designing plugin-style architectures common in ML pipelines.

```python
from abc import ABC, abstractmethod
 
class Detector(ABC):
    @abstractmethod
    def detect(self, image):
        \"\"\"Return list of bounding boxes.\"\"\"
        pass
 
    @abstractmethod
    def load_weights(self, path):
        pass
 
    def visualize(self, image, boxes):
        \"\"\"Concrete method shared by all detectors.\"\"\"
        for box in boxes:
            x1, y1, x2, y2 = box["coords"]
            # draw rectangle on image
        return image
 
class YOLODetector(Detector):
    def detect(self, image):
        # YOLO-specific detection logic
        return [{"coords": (10, 20, 100, 200), "class": "car", "conf": 0.95}]
 
    def load_weights(self, path):
        self.model = None  # load YOLO weights
 
# detector = Detector()         -> TypeError: cannot instantiate abstract class
# detector = YOLODetector()     -> works fine

```

## 1.2 Python Memory Model

Understanding how Python manages memory is a frequent senior interview topic. Python uses reference counting as its primary garbage collection mechanism, supplemented by a cyclic garbage collector.

### 1.2.1 Reference Counting and Identity

Every Python object has a reference count. When it reaches zero, the memory is freed. The `id()` function returns the memory address of an object, and the `is` keyword checks if two references point to the same object.

```python
import sys
 
a = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (a + temporary ref from getrefcount)
 
b = a                      # b points to the same list object
print(a is b)              # True  -> same object in memory
print(id(a) == id(b))      # True
 
c = [1, 2, 3]              # c is a NEW list with same values
print(a == c)              # True  -> values are equal
print(a is c)              # False -> different objects
 
# Mutable default argument trap:
def bad_append(item, lst=[]):    # The default list is shared across calls!
    lst.append(item)
    return lst
 
print(bad_append(1))   # [1]
print(bad_append(2))   # [1, 2]  -> BUG! Same list object
 
# Correct pattern:
def good_append(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

```

### 1.2.2 Mutable vs Immutable Objects

Immutable objects: `int`, `float`, `str`, `tuple`, `frozenset`, `bytes`.
Mutable objects: `list`, `dict`, `set`, `bytearray`, custom classes.
This distinction affects function arguments, dictionary keys, and copying behavior.

```python
# Immutable: operations create new objects
x = "hello"
y = x
x = x + " world"     # creates a new string
print(y)              # "hello" -> unchanged
 
# Mutable: modifications affect all references
a = [1, 2, 3]
b = a
a.append(4)
print(b)              # [1, 2, 3, 4] -> b changed too!
 
# This is why tuples can be dict keys but lists cannot:
d = {(1, 2): "point"}     # OK
# d = {[1, 2]: "point"}   # TypeError: unhashable type: "list"

```

## 1.3 Deep Copy vs Shallow Copy

Shallow copy creates a new object but inserts references to the nested objects. Deep copy recursively copies all nested objects, creating a completely independent clone.

```python
import copy
 
# Shallow copy: top-level is new, nested objects are shared
original = {"model": "yolo", "params": [0.1, 0.2, 0.3], "meta": {"v": 1}}
shallow = copy.copy(original)
 
shallow["model"] = "ssd"            # does NOT affect original
shallow["params"].append(0.4)       # AFFECTS original! (shared list)
shallow["meta"]["v"] = 2            # AFFECTS original! (shared dict)
 
print(original["params"])  # [0.1, 0.2, 0.3, 0.4]
print(original["meta"])    # {"v": 2}
 
# Deep copy: everything is independent
original2 = {"model": "yolo", "params": [0.1, 0.2, 0.3], "meta": {"v": 1}}
deep = copy.deepcopy(original2)
 
deep["params"].append(0.4)          # does NOT affect original2
deep["meta"]["v"] = 2               # does NOT affect original2
 
print(original2["params"])  # [0.1, 0.2, 0.3]
print(original2["meta"])    # {"v": 1}
 
# Custom __copy__ and __deepcopy__:
class Pipeline:
    def __init__(self, steps):
        self.steps = steps
        self._cache = {}
 
    def __deepcopy__(self, memo):
        \"\"\"Deep copy steps but reset cache.\"\"\"
        new = Pipeline(copy.deepcopy(self.steps, memo))
        new._cache = {}  # fresh cache
        return new

```

## 1.4 Dunder (Magic) Methods

Dunder methods let your classes integrate with Python built-in operations like `len()`, `str()`, iteration, arithmetic, comparison, and context managers. A senior engineer is expected to know these thoroughly.

### 1.4.1 Core Dunder Methods

```python
class BoundingBox:
    \"\"\"A bounding box for object detection results.\"\"\"
 
    def __init__(self, x1, y1, x2, y2, label="", confidence=0.0):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.label = label
        self.confidence = confidence
 
    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
 
    # String representations
    def __repr__(self):
        return (f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2}, "
                f"label='{self.label}', confidence={self.confidence:.2f})")
 
    def __str__(self):
        return f"[{self.label}] ({self.x1},{self.y1})->({self.x2},{self.y2}) conf={self.confidence:.2f}"
 
    # Comparison (by confidence)
    def __lt__(self, other):
        return self.confidence < other.confidence
 
    def __eq__(self, other):
        return (self.x1 == other.x1 and self.y1 == other.y1 and
                self.x2 == other.x2 and self.y2 == other.y2)
 
    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2))
 
    # Container protocol
    def __len__(self):
        return 4  # number of coordinates
 
    def __getitem__(self, idx):
        return (self.x1, self.y1, self.x2, self.y2)[idx]
 
    # Arithmetic: IoU via & operator
    def __and__(self, other):
        \"\"\"Calculate IoU between two boxes.\"\"\"
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0
 
    # Context manager
    def __enter__(self):
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

```

### 1.4.2 Important Dunder Methods Reference

| Method | Triggered By | Purpose |
| --- | --- | --- |
| `__init__` | `ClassName()` | Initialize instance |
| `__repr__` | `repr(obj)` | Unambiguous string for debugging |
| `__str__` | `str(obj)`, `print()` | Human-readable string |
| `__len__` | `len(obj)` | Return length/size |
| `__getitem__` | `obj[key]` | Index/key access |
| `__setitem__` | `obj[key] = val` | Index/key assignment |
| `__contains__` | `x in obj` | Membership test |
| `__iter__` | `for x in obj` | Return iterator |
| `__next__` | `next(iterator)` | Get next item |
| `__call__` | `obj()` | Make instance callable |
| `__enter__`/`__exit__` | `with obj:` | Context manager protocol |
| `__add__` | `obj + other` | Addition operator |
| `__eq__`/`__lt__` | `==`, `<` | Comparison operators |
| `__hash__` | `hash(obj)` | Hash for sets/dict keys |

## 1.5 Decorators

A decorator is a function that takes a function (or class) and returns a modified version. They are syntactic sugar for wrapping functions. Understanding closures is prerequisite.

### 1.5.1 Function Decorators

```python
import functools
import time
import logging
 
# Basic decorator with wraps
def timer(func):
    @functools.wraps(func)  # preserves __name__, __doc__
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper
 
@timer
def train_epoch(model, dataloader):
    \"\"\"Train one epoch.\"\"\"
    pass
 
# Decorator with arguments
def retry(max_attempts=3, delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
                    print(f"Retry {attempt + 1}/{max_attempts}: {e}")
        return wrapper
    return decorator
 
@retry(max_attempts=5, delay=2.0)
def download_model(url):
    \"\"\"Download with retries.\"\"\"
    pass
 
# Caching decorator (like lru_cache but custom)
def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    wrapper.clear_cache = cache.clear
    return wrapper

```

### 1.5.2 Class Decorators

```python
# Decorator that adds a registry pattern
model_registry = {}
 
def register_model(name):
    def decorator(cls):
        model_registry[name] = cls
        return cls
    return decorator
 
@register_model("yolov8")
class YOLOv8:
    pass
 
@register_model("faster_rcnn")
class FasterRCNN:
    pass
 
# Usage: model = model_registry["yolov8"]()
 
# Singleton pattern via decorator
def singleton(cls):
    instances = {}
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance
 
@singleton
class ModelManager:
    def __init__(self):
        self.models = {}

```

## 1.6 Iterators and Generators

Iterators implement the `__iter__` and `__next__` protocol. Generators are a concise way to create iterators using the `yield` keyword. They are memory-efficient because they produce values one at a time (lazy evaluation), which is essential for processing large datasets.

### 1.6.1 Custom Iterator

```python
class ImageBatchIterator:
    \"\"\"Iterate over images in batches, loading from disk on demand.\"\"\"
 
    def __init__(self, image_paths, batch_size=32):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self._index = 0
 
    def __iter__(self):
        self._index = 0
        return self
 
    def __next__(self):
        if self._index >= len(self.image_paths):
            raise StopIteration
        batch_paths = self.image_paths[self._index:self._index + self.batch_size]
        self._index += self.batch_size
        return [self._load_image(p) for p in batch_paths]
 
    def _load_image(self, path):
        return path  # placeholder for actual image loading
 
    def __len__(self):
        import math
        return math.ceil(len(self.image_paths) / self.batch_size)

```

### 1.6.2 Generators

```python
# Generator function
def frame_generator(video_path, skip=1):
    \"\"\"Yield frames from a video, one at a time (memory efficient).\"\"\"
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()
 
# Usage: for idx, frame in frame_generator("video.mp4", skip=5):
 
# Generator expression (like list comprehension but lazy)
file_sizes = (os.path.getsize(f) for f in os.listdir("."))
# total = sum(file_sizes)  -> never holds all sizes in memory at once
 
# yield from: delegate to sub-generator
def all_detections(image_folder):
    for subfolder in os.listdir(image_folder):
        yield from detect_in_folder(os.path.join(image_folder, subfolder))
 
# send() to generators:
def adaptive_threshold():
    \"\"\"Generator that adjusts threshold based on sent feedback.\"\"\"
    threshold = 0.5
    while True:
        result = yield threshold
        if result is not None:
            threshold = result  # caller can adjust

```

## 1.7 LeetCode Practice Strategy

Spend 1-1.5 hours daily on LeetCode easy-medium problems in Python. Focus on these patterns which appear most frequently in ML engineering interviews:

| Pattern | Key Problems | Difficulty | ML Relevance |
| --- | --- | --- | --- |
| **Hash Maps** | Two Sum, Group Anagrams | Easy-Medium | Feature lookup, caching |
| **Sliding Window** | Max Subarray, Min Window | Medium | Temporal features |
| **Binary Search** | Search Rotated Array | Medium | Threshold finding |
| **BFS/DFS** | Number of Islands | Medium | Connected components |
| **Dynamic Prog.** | Longest Common Subseq | Medium | Sequence alignment |
| **Sorting** | Merge Intervals | Medium | NMS, bbox merging |
| **Heap/Priority Q** | Top K Frequent | Medium | Top-K predictions |

"""

with open("python_for_ml.md", "w", encoding="utf-8") as f:
f.write(content)

```
Your markdown file is ready
[file-tag: code-generated-file-0-1778929050765120433]

I have formatted the information you provided into a clean Markdown document (`python_for_ml.md`). The code snippets have been wrapped in Python syntax highlighting blocks, and the LeetCode practice strategy has been transformed into a well-structured table for easy reading. Let me know if you need any edits or additions!

```