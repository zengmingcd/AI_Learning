# Azure AI Fundamentals Certification 2024 (AI-900) - Full Course to PASS the Exam
### Level: Beginner
### Link: [YouTube](https://youtu.be/hHjmr_YOqnU?si=mWBipfvpflzMuoT4)
### Link: [Website: Exampro](https://app.exampro.co/student/journey/AI-900)
### Duration: 15 Hours
---

# Course Note
## Introduction

## ML and AI Concepts
### Layer of ML
- What is Artificial Intelligence (AI)?
  - Machines that perform jobs that mimic(模仿) human behavior.
- What is Machine Learning (ML)?
  - Machines that get better at a task without explicit programming(显式编程).
- What is Deep Leaning (DL)?
  - Machines that have an artificial neural network inspired by the human brain to solve complex problems.
- What is Data Scientist?
  - A person with multi-disciplinary(多学科) skills in math, statistics, predictive modeling and machine learning to make future predictions.

- AI>ML>DL

### Key Elements of AI
- AI is the software that imitates human behaviors and capabilities
- Key Elements (According to Microsoft/Azure)
  - Machine Learning - the foundation of an AI system, learn and predict like a human.
  - Anomaly detection(异常检测) - detect outliers or things out of place like a human. 像人类一样检测异常值或不合适的事物。
  - Computer Vision - be able to see like a human
  - Natural Language Processing - be able to process human languages and infer content(推断内容).
  - Conversational AI - be able to hold a conversation with a human.

### DataSets
- What is a dataset?
  - A data set is a logical grouping of units of data that are closely related and/or share the same data structure.
- There are publicly available data sets that used in the learning of statistics, data analytics, machine learning.
  - MNIST database
    - Image of handwritten digits used to test classification, clustering, and image processing algorithm.
    - Commonly used when learning how to build computer vision ML models to translate handwriting into digital text.
  - Common Objects In Context(COCO) dataset
    - A dataset which contains many common images using a JSON file (coco format) that identify objects or segments within an image.
    - features:
      - Object segmentation.(对象分割)
      - Recognition in context
      - Superpixel stuff segmentation
      - 329K images (>200K labeled)
      - 0.5 million object instance
      - 79 object categories
      - 90 stuff categories
      - 4 captions per image
      - 249000 people with keypoints

### Labeling
- What is Data Labeling?
  - the process of identifying raw data(原始数据) (images, text file, videos, etc.) and adding one or more meaningful and informative labels to provide context so that a machine leaning model can learn.
  - With supervised machine learning, labeling is a prerequisite(前提条件) to produce training data and each pieces of data will generally be labeled by a human.
  - With unsupervised machine learning, labels will be produced by machine, and may not be human readable.
- What is ground truth(基本事实)?
  - A properly labeled dataset that you use as the objectives standard to train and access a given model is often called "ground truth". The accuracy of your trained model will depend on the accuracy of your ground truth.

### Supervised Learning & Unsupervised Learning & Reinforcement Learning
- Supervised Learning (SL)
  - Data that has been labeled for training
  - Task-driven - make a prediction
  - When the labels are known and you want a precise(准确的) outcome.
  - When you need a specific value return.
  - eg. Classification, Regression
- Unsupervised Learning (SL)
  - Data has not been labeled, the ML model needs to do its own labeling.
  - Data-driven - recognize a structure or pattern
  - When the labels are not known and the outcome does not need to be precise.
  - When you are trying to make sense of data.
  - eg. Clustering, Dimensionality Reduction(降维), Association(关联)
-  Reinforcement Learning (RI)
   -  There is no data, there is an environment and an ML model generates data any many attempt(尝试) to reach a goal.
   -  Decisions-driven - Game AI, Learning Tasks, Robot Navigation.

### Neural Networks & Deep Learning
- What are Neural Networks? (NN)
  - Often described as mimicking(模仿) the brain, a neuron/node represents an algorithm.
  - Data is inputted into a neuron and based on the output the data will be passed to one of many other connected neural.
  - The connection between neurons is weighted(权重).
  - The network is organized in layers.
  - There will be a input layer, 1 to many hidden layers and an output layer.
- What is Deep Learning?
  - A neural network that has 3 or more hidden layers is considered deep learning.
- What is Feed Forward? (FNN)
  - Neural Networks where connections between nodes do not form a cycle. (always move forward)
- What is Backpropagation? (BP)
  - Moves backwards through the neural network adjusting weights to improve outcome on next iteration. This is how a neural net learns.
- What is Loss Function
  - A function that compares the ground truth to the prediction to determine the error rate (how bad the network performed)
- What is Activation Functions(激活函数)
  - An algorithm applied to a hidden layer node that affects connected output.
  - eg. ReLu
- What is Dense(稠密)
  - When the next layer increases the amount of nodes.
- What is Sparse(稀疏)
  - When the next layer decreases the amount of nodes.

### GPU
- What is a GPU
  - A General Processing Unit(GPU) that is specially designed to quickly render(渲染) high-resolution images and video concurrently.
- GPUs can perform parallel operations(并行操作) on multiple set of data, and so they are commonly used for non-graphical tasks such as machine learning and scientific computation(科学计算).
  - CPU can have average 4 to 16 processor cores
  - GPU can thousands of processor cores
  - 4 to 8 GPUs can provide as many as 40,000 cores.
- GPUs are best suited for repetitive(重复) and highly-parallel(高并发) computing tasks
  - Rendering graphics
  - Cryptocurrency(加密货币) mining
  - Deep Learning and Machine Learning

### CUDA (不考)
- What is NVIDIA
  - NVIDIA is a company that manufactures graphical processing units (GPUs) for gaming and professional markets.
- What is CUDA
  - Compute Unified Device Architecture(CUDA) is a parallel computing platform and API by NVIDIA that allows developers to use CUDA-enable GPUs for general-purpose computing on GPUs(GPGPU)
    - All major deep learning frameworks are integrated with NVIDIA Deep Learning SDK.
  - The NVIDIA Deep Learning SDK is a collection of NVIDIA libraries for deep learning.
  - One of those libraries is the CUDA Deep Neural Network library(cuDNN)
  - cuDNN provides highly tuned implementations for standard routines such as:
    - Forward and backward convolution(向前及向后卷积)
    - Pooling(池)
    - Normalization(标准化)
    - Activation layers(激活层)

### Machine Learning Pipeline
![ML Pipeline](MLPipeline.png)
- Data Labeling
  - For supervised learning you need to label your data so the ML model can learn by example during training
- Feature Engineering
  - ML models only work with numerical data. 
  - So you need to translate it into a format that it can understand, extract out the important data that the ML needs to focus on.
- Training
  - Your model needs to learn how to become smarter. 
  - It will perform multiple iterations getting smarter with each iteration.
- Hyperparameter Tunning (超参数调优)
  - An ML model can have different parameters, we can use ML to try out many different parameters to optimize the outcome.
  - When you touch the deep learning, it is impossible to track the parameters by hand, you have to use hyperparameter tunning.
- Serving(deploy as a service)
  - We need to make our ML model accessible, so we serve by hosting in a virtual machine or container.
- Inference(推理)
  - Inference is the act of requesting to make a prediction.
  - Including:
    - Real-time Endpoint
    - Batch Processing

### Forecasting & Prediction
- What is a Forecasting? (预报)
  - Make a future prediction with relevant data
    - Analysis of trends
    - It's not "guessing"
- What is a Prediction? (预测)
  - Make a future prediction without relevant data
    - Use statistics to predict future outcomes
    - It's more of "guessing"
    - Uses decision theory(决策理论).

### Metrics
- What is Metrics
  - Performance/Evaluation Metrics are used to evaluate different Machine Learning Algorithms.
  - For different types of problems different metrics matter,(this is not an exhaustive list)
    - Classification Metrics
      - accuracy
      - precision
      - recall
      - F1-score
      - ROC
      - AUC
    - Regression Metrics
      - MSE
      - RMSE MAE
    - Ranking Metrics
      - MRR
      - DCG
      - NDCG
    - Statistical Metrics
      - Correlation
    - Computer Vision Metrics
      - PSNR
      - SSIM
      - IoU
    - NLP Metrics
      - Perplexity
      - BLEU
      - METEOR
      - ROUGE
    - Deep Learning Related Metrics
      - Inception score
      - Frechet Inception distance
  - There are two categories of evaluation metrics
    - Internal Evaluation - metrics used to evaluate the internals of the ML model
      - The famous four used in all kinds of models
        - Accuracy
        - F1 score
        - Precision
        - Recall
    - External Evaluation - metrics used to evaluate the final prediction of the ML model.

### Jupyter Notebook & JupyterLab
- What is Jupyter Notebook
  - A Web-based application for authoring(创作) documents that combine:
    - Live-code
    - Narrative text(叙述文字)
    - Equations(方程式)
    - Visualizations
  - iPython's notebook feature became Jupyter Notebook
  - Jupyter Notebooks were overhauled(大修，重构) and better integrated into an IDE called JupyterLab
    - You generally want to open Notebooks in Labs
    - The legacy(遗留的) web-based interface is known as Jupyter classic notebook

- What is JupyterLab
  - JupyterLab is a next-generation web-based user interface
  - All the familiar features of the classic Jupyter Notebook in a flexible and powerful user interface:
    - Notebook
    - Terminal
    - Text editor
    - File browser
    - Rich outputs
  - JupyterLab will eventually replace the classic Jupyter Notebook.


### Regression
- What is Regression
  - Regression is a process of finding a function to correlate(关联) a label dataset into continuous variable/number(连续变量/连续数字).
- Outcome: Predict this variable in the future.
- Theory:
  - Vectors(dots) are plotted on a graph in multiple dimensions, eg (X,Y)
  - A regression line is drawn though the dataset.
- Algorithm:
  - The distance of the vector from the regression line called an Error.
  - Different Regression Algorithms use the error to predict future variables:
    - Mean squared error (MSE)
    - Root mean squared error (RMSE)
    - Mean absolute error (MAE)
  
### Classification
- What is Classification
  - Classification is a process of finding a function to divide a labeled dataset into class/categories
- Outcome: Predict category to apply to the input data.
- Theory:
  - Vectors(dots) are plotted on a graph in multiple dimensions, eg (X,Y)
  - A classification line divides the dataset
- Algorithm:
  - Logistic Regression
  - Decision Tree/Random Forest
  - Neural Networks
  - Naive Bayes
  - K-Nearest Neighbors(KNN)
  - Support Vector Machines

### Clustering
- What is Clustering
  - Clustering is a process grouping unlabeled data base on similarities and differences.
- Outcome: Group data based on their similarities or differences.
- Theory:
  - Vectors(dots) are plotted on a graph in multiple dimensions, eg (X,Y)
  - Group the dataset based on their similarities or differences.
- Algorithm:
  - K-means
  - K-medoids
  - Density Based
  - Hierarchical
  
  ### Confusion Matrix
  - What is Confusion Matrix
    - A confusion matrix is table to visualize the model predictions(predicted) vs ground truth labels(actual).
    - Also known as an error matrix. They are useful in classification problems.
  - The size of matrix is dependent on the labels: 
    - Labels * 2 (Yes/No)
  - Indicator:
    - Actual No: false
    - Actual Yes: true
    - Predict No: negative
    - Predict Yes: positive
    - Actual No + Predict No = False Negative(FN)
    - Actual No + Predict Yes = False Positive(FP)
    - Actual Yes + Predict No = True Negative(TN)
    - Actual Yes + Predict Yes = True Positive(TP)
    - FN + FP = Total False (tF) 实际为No的总数
    - TN + TP = Total True (tT) 实际为Yes的总数
    - FN + TN = Total Negative (tN) 预测为No的总数
    - FP + TP = Total Positive (tP) 预测为Yes的总数
    - tF + tT = tN + tP = Total 总数

### 
