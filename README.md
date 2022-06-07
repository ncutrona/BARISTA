# RPNB-Framework
Regularized Proximal Descent-Based Naive Bayes Framework


**About the Framework**

The RPNB Framework is a class-specific attribute weighted naive Bayes (CAWNB) method that induces an L1 penalty on the weights. With the use of proximal descent, an efficient optimization algorithm, the paramters of the framework can be learned without a demanding computational overhead. Class-specific attribute weights help alleviate the conditional independence assumption of naive bayes, produce better discriminative power, and provide a feature importance mapping of the data. With a large feature space; however, the amount of paramters to learn increases drastically. In order to avoid overfitting, frameworks have been proposed to induce a penalty on the weights. This framework proposes to induce an L1 penalty on the class-specific attribute weights. Optimization techniques such as the sub-gradient are inefficient and require too much computation to learn efficiently. We propose to induce an L1 penalty while using *proximal gradient descent*, an efficient and novel approach to the CAWNB framework for regularization and model learning.

*FRAMEWORK PROCESSES*:

<img width="1140" alt="Screen Shot 2022-06-07 at 1 15 43 PM" src="https://user-images.githubusercontent.com/59042355/172443629-b1ae684d-a9f7-4a63-8349-8e3736efd821.png">


**Implementation**

Users should use this framework for classifciation tasks with large feature spaces. This model is designed to handle a large number of features through regularization, while still yielding strong discriminatory power through the class specific attribute weights. This framework also produces interpretability around classifications because of the probabilistic nature of the model.  


**How to Use**

In order to use this framework, simply clone the repo to your local machine and run the notebook file (model.ipynb). Instructions are given in the notebook file on how to select hyper paramters and load in your data set. The framework assumes all pre-processing has been done prior to model learning.


**Working Paper**

Title: WIP

Abstract: WIP

Link: WIP
