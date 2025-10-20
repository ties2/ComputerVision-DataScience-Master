# Data annotation

Data annotation is the process of labeling or tagging data to make it usable for training Machine Learning (ML) models, particularly in supervised learning. It involves applying meaningful tags, labels, attributes, or metadata to raw data (such as images, text, audio, or video) to clarify what that data represents.


The resulting labeled dataset serves as the ground truth that the ML model learns from.

Key Aspects of Data Annotation
1. Types of Data Annotated

|Data Type	|Annotation Example	|ML Application |
|---- | ---- | ---- |
|Images/Video	|Bounding boxes, polygons, keypoints, semantic masks.	|Object Detection, Autonomous Driving.
|Text	|Named entity recognition (NER), sentiment labeling, summarization.	|Natural Language Processing (NLP), Chatbots.
|Audio	|Transcription, speaker identification, emotion tagging.	|Speech Recognition, Voice Assistants.
|Sensor/Time-Series	|Labeling periods of activity, anomalies, or specific states.	|Predictive Maintenance, Healthcare Monitoring.|

2. Common Annotation Techniques
The choice of technique depends on the ML task:

|Technique	|Description	|Use Case
| ---- | ---- | ---- 
|Classification	|Assigning a single tag to the whole data unit.	|"This image contains a cat."
|Bounding Box	|Drawing rectangular boxes around specific objects.	|Locating cars, pedestrians, or products in a scene.
|Semantic Segmentation	|Labeling every pixel in an image with a class label (e.g., road, sky, car).	|Understanding scene geometry for robotics.
|Polygonal Segmentation	|Using precise, multi-sided shapes to delineate object boundaries.	|Identifying irregular shapes like tumors or geographical areas.
|Keypoint/Landmark	|Pinpointing specific coordinates on an object.	|Facial recognition (eyes, nose), pose estimation (joints).
|Named Entity Recognition (NER)	|Identifying and categorizing specific entities (Person, Location, Organization) within text.	|Extracting information from documents.