# Project List: Computer Vision and Data Science

Welcome to the Projects section of the Computer Vision and Data Science Learning Hub! 
This file provides a curated list of GitHub repositories, each dedicated to a specific area of study or task in computer vision and data science.
These projects contain code, notebooks, datasets, and documentation to help you learn and apply concepts from educational journey.

Whether you're exploring foundational topics or diving into advanced pipelines, these repositories are designed to be practical and accessible.
FundamentalsDive into the core building blocks of computer vision, including Linear Algebra, Image Features, Image Filtering, Annotation, Validation, 
Data Exploration, Dimensionality Reduction, and Image Transforms.

* Beginner Level (Core Skills)

Focus: Data wrangling, visualization, basic ML algorithms, statistical analysis.
Skills: Python, Pandas, NumPy, Matplotlib, Scikit-learn, A/B testing.

* Intermediate Level (ML and Deep Learning)

Focus: Classification, regression, neural networks, NLP, recommendation systems.
Skills: TensorFlow/Keras, PyTorch, NLP libraries (NLTK, SpaCy), recommendation algorithms.

* Advanced Level (Real-Time and Scalable Systems):

Focus: Real-time systems, MLOps, distributed systems, reinforcement learning, AutoML.
Skills: OpenCV, MLflow, Kubernetes, Ray, reinforcement learning frameworks.

* Specialized/Research-Oriented: 

Focus: Language models, generative AI.
Skills: Transformers, Hugging Face, GANs.

### sample pipeline ML

```
project_root/
├── README.md               # Project overview, setup instructions, and usage.
├── LICENSE                 # License file (e.g., MIT, Apache).
├── .gitignore              # Git ignore file for temporary files, datasets, etc.
├── requirements.txt        # List of Python dependencies (e.g., pip install -r requirements.txt).
├── setup.py                # For packaging the project as a Python module (optional for advanced setups).
├── Dockerfile              # For containerizing the application.
├── docker-compose.yml      # For multi-container setups (e.g., with databases).
├── config/                 # Configuration files.
│   ├── config.yaml         # Main config for hyperparameters, paths, etc. (use YAML for flexibility).
│   └── secrets.yaml        # Sensitive info like API keys (git-ignored).
├── data/                   # Data storage (often git-ignored for large files; use DVC for versioning).
│   ├── raw/                # Raw, unprocessed data files (e.g., CSV, JSON, images).
│   ├── processed/          # Cleaned and preprocessed data.
│   ├── external/           # Third-party data sources.
│   └── interim/            # Temporary data during processing.
├── docs/                   # Documentation.
│   ├── api.md              # API documentation if deploying as a service.
│   └── architecture.md     # High-level pipeline diagrams (e.g., using Markdown or PlantUML).
├── notebooks/              # Jupyter notebooks for exploration and prototyping.
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                    # Source code for the pipeline.
│   ├── __init__.py         # Makes src a Python package.
│   ├── data/               # Data-related scripts.
│   │   ├── ingest.py       # Script for data ingestion (e.g., from APIs, databases).
│   │   └── preprocess.py   # Data cleaning and transformation.
│   ├── features/           # Feature engineering.
│   │   └── build_features.py # Feature extraction and selection.
│   ├── models/             # Model-related code.
│   │   ├── train.py        # Training script with hyperparameter tuning (e.g., using GridSearchCV or Optuna).
│   │   ├── predict.py      # Inference script for predictions.
│   │   └── evaluate.py     # Evaluation metrics and reporting.
│   ├── deployment/         # Deployment scripts.
│   │   ├── app.py          # Flask/FastAPI app for serving the model.
│   │   └── deploy.sh       # Bash script for deployment (e.g., to AWS, GCP).
│   ├── utils/              # Utility functions.
│   │   ├── logging.py      # Custom logging setup.
│   │   └── helpers.py      # General helpers (e.g., data loaders).
│   └── pipeline.py         # Main orchestration script to run the entire pipeline.
├── tests/                  # Unit and integration tests.
│   ├── test_data.py        # Tests for data processing.
│   ├── test_models.py      # Tests for model training and evaluation.
│   └── test_utils.py       # Tests for utilities.
├── models/                 # Saved models and artifacts (git-ignored; version with MLflow or DVC).
│   ├── trained_model.pkl   # Serialized model (e.g., via joblib or pickle).
│   └── model_metadata.yaml # Model version, metrics, etc.
├── reports/                # Output reports and visualizations.
│   ├── figures/            # Plots (e.g., confusion matrices, ROC curves).
│   └── metrics.json        # JSON file with evaluation results.
├── .dvc/                   # DVC configuration for data versioning (if using DVC).
├── mlflow/                 # MLflow tracking artifacts (if using MLflow for experiment tracking).
└── workflows/              # CI/CD and orchestration workflows.
    ├── .github/workflows/  # GitHub Actions for CI/CD.
    │   └── ci-cd.yaml      # YAML for automated testing and deployment.
    └── airflow_dags/       # DAGs if using Apache Airflow for pipeline orchestration.
```


Mathematics for Machine LearningExplore essential mathematical concepts like linear algebra, calculus, and probability, with practical applications 
to machine learning algorithms.

📚 Project Repositories
Below is a list of projects, each hosted in its own GitHub repository:

| title | info | 
| ---------- | ----------
| [Computer-vision-yolo8-project](https://github.com/ties2/Computer-vision-yolo8-project) | computer vision project| 
| [Adult_Income_Prediction_NN](https://github.com/ties2/Adult_Income) | NN project [kaggle](https://www.kaggle.com/code/nirvanafl/adult-income-prediction-nn)| 



Scientific Programming

Scientific ProgrammingLearn to use tools like Python, NumPy, OpenCV, PyTorch, and TensorFlow to write clean, efficient, and reproducible code for data science and computer vision.

Hardware and 3D

Hardware and 3DUnderstand the hardware behind computer vision, covering Computing Hardware (1 & 2), Camera Geometry (1 & 2), 3D Computer Vision, Vision Hardware, and Spectral Imaging.

Learning Methodologies

Learning MethodologiesMaster machine learning techniques, including Statistical Models, Linear Models, Neural Networks, Convolutional Neural Networks, Temporal Neural Networks, Hyperparameter Tuning (1 & 2), Synthetic Data, and Advanced Architectures.

Tasks (Pipelines)

My First PipelineA beginner-friendly pipeline to get started with computer vision tasks, perfect for newcomers.

ClassificationExplore image classification techniques, covering both introductory (Classification 1) and advanced (Classification 2) methods.

SegmentationLearn image segmentation, from basic approaches (Segmentation 1) to advanced techniques (Segmentation 2).

Object DetectionDiscover object detection methods, including foundational (Object Detection 1) and advanced (Object Detection 2) approaches.

Unsupervised LearningDive into unsupervised learning techniques like clustering and autoencoders.

Anomaly DetectionLearn to identify outliers and anomalies in data using specialized methods.

Explainable AIUnderstand how to interpret and explain AI model decisions for transparency and trust.


Ethics and Professional Skills

Ethics and Professional SkillsExplore ethical considerations (Ethics 1 & 2) and scientific writing (Scientific Writing 1 & 2) for responsible AI and data science practices.

Miscellaneous and Future Directions

Reinforcement LearningLearn about reinforcement learning for decision-making in dynamic environments.

Big DataTackle large-scale data processing and analysis techniques.

Data-centric AIFocus on improving data quality and preparation for better machine learning outcomes.


How to Use These Projects

Visit a Repository:

Click any of the links above to access a project’s GitHub repository.
Each repository includes a README.md with instructions, code, and resources specific to that topic.


Clone a Repository:

Example for cloning the Fundamentals project:git clone https://github.com/ties2/computer-vision-fundamentals.git

Install Dependencies:

Most projects use Python. Check each repository’s requirements.txt for dependencies:pip install -r requirements.txt




Start Exploring:

Begin with My First Pipeline for an easy introduction.
Move to advanced topics like Convolutional Neural Networks or Object Detection for deeper learning.



🔗 Additional Resources

Check the main repository’s resources/ folder for datasets, research papers, and external teaching materials.
Visit the lectures/ folder for kick-off presentations and supplementary content.

💡 Contributing
Want to contribute to these projects? Suggestions, code improvements, or additional resources are welcome!

Fork the relevant project repository.
Create a new branch (git checkout -b feature-branch).
Submit a pull request with a clear description of your changes.

📬 Contact
Have questions or feedback? Reach out:

GitHub Issues: Create an issue
Email: nirvana.elahi@outlook.com


