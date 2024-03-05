# Fundamentals of Machine Learning
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/fundamentals-machine-learning/)
### Duration: 3 Hours
---

## Course Note
### What is machine learning
- Machine learning has its origins in statistics and mathematical modeling of data. 机器学习起源于数据的统计和数学建模。
- The fundamental idea of machine learning is to use data from past observations to predict unknown outcomes or values. 机器学习的基本思想是使用过去观察的数据来预测位置的结果或值。
- A machine learning model is a software application that encapsulates a function to calculate an output value based on one or more input values. 机器学习是一种软件应用，它封装了根据一个或多个输入值计算出值的“函数”。
- The process of defining that function is known as training. 定义函数的过程就叫“训练”。
- After the function has been defined, you can use it to predict new values in a process called inferencing. 模型训练好后，使用模型预测新值的过程叫“推理”。
  
### Steps involved in training and inferencing
1. The training data consists of past observations. 将过去观察结果作为训练的数据
   - In most cases, the observations include the observed attributes or features of the thing being observed, and the known value of the thing you want to train a model to predict (known as the label) 多数情况下观察结果包括：被观察实物的属性或特征，想要模型预测的已知结果（就是“标签”）
2. An algorithm is applied to the data to try to determine a relationship between the features and the label, and generalize that relationship as a calculation that can be performed on x to calculate y. 算法使用数据来常识确定特征和标签之间的关系，并将该关系概括为对X进行计算，得到Y的计算。
   - the basic principle is to try to fit the data to a function in which the values of the features can be used to calculate the label. 基本原则是尝试“拟合”数据得到一个函数，这个函数可以用特征值来计算标签值。
3. The result of the algorithm is a model that encapsulates the calculation derived by the algorithm as a function f. 算法的结果是一个模型，即将算法导出的计算方法封装为“函数”。
4. The training phase is complete, the trained model can be used for inferencing. 训练阶段完成，训练后的模型就可以用于推理了。
   - The model is essentially a software program that encapsulates the function produced by the training process. 模型本质上是一个封装了训练过程产生的函数的软件程序。
   - You can input a set of feature values, and receive as an output a prediction of the corresponding label. 你可以输入一组特征值，并接受相应标签的预测作为输出。
   - Because the output from the model is a prediction that was calculated by the function, and not an observed value, you'll often see the output from the function shown as ŷ (which is rather delightfully verbalized as "y-hat").输出的是预测值，而不是观察值，所以不用y表示，用ŷ表示


### Types of machine learning
![Type of ML](image.png)

- Supervised machine learning 监督学习
  - A general term for machine leaning algorithms in which the training data include both feature values and known label values. 学习数据包括特征值和已知标签值的机器学习算法一类。
  - Types of Supervised ML
    - Regression. 回归
      - A form of supervised ML in which the label predicted by the model is numeric value. 模型预测的标签是一个数值。
    - Classification. 分类
      - A form of supervised ML in which the label represents a categorization or class. 标签代表分类别或类
      - Types of Classification
        - Binary classification. 二元分类
          - The label determines whether the observed item is (or isn't) an instance of a specific class. 标签确定被观察项目是或不是一个特定类的实例。
          - Predict one of two mutually exclusive outcomes. 预测两个互斥的结果。
        - Multiclass classification 多元分类
          - Extends binary classification to predict a label that represents one of multiple possible classes. 扩展了二元分类以预测表示多个互斥可能类别的标签。

- Unsupervised machine learning 无监督学习
  - Involves training models using data that consists only of feature values without any known labels. 训练数据仅包含特征值，不包含已知标签值。
  - Determine relationships between the features of the observations in the training data. 确定训练数据中被观察项目之间的关系。
  - Types of Unsupervised ML
    - Clustering 聚类
      - Identifies similarities between observations based on their features, and groups them into discrete clusters. 根据被观察项目的特征识别出观察值的关系，并将其放到离散的簇分组中。