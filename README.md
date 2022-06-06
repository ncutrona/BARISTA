# RPNB-Framework
Regularized Proximal Descent-Based Naive Bayes Framework


**About the Framework**

The RPNB Framework is a class-specific attribute weight naive Bayes method that induces an L1 penalty on the learned weights. With the use of proximal descent, an efficient optimization algorithm, the paramters of the framework can learned without a demanding computational overhead. Class-specific attribute weights help alleviate the conditional independence assumption of naive bayes, produce better discriminative power, and provide a feature importance mapping of the data. With a large feature space; however, the amount of paramters to learn increases drastically. In order to avoid overfitting, frameworks have been proposed to induce a penalty on the weights. This framework proposes to induce an L1 penalty on the class-specific attribute weights. Optimization techniques such as the sub-gradient are inefficient and require to much computation to learn. We propose to still induce an L1 penalty, but use *proximal gradient descent*, an efficient and novel approach for this framework.

*Some Computation to be familiar with*:

**Bayes Rule**: p(c ┤|  x)=  (p(x ┤|  c)P(c))/(p(x))![image](https://user-images.githubusercontent.com/59042355/172247710-2e26832e-fd6a-435c-95c9-61b4400448bb.png)
 
