# Bayesian Regularized Iterative Soft Thresholding Algorithm


### Paper Abstract

Weighted Naive Bayes methods have recently been devel-
oped to alleviate the strong conditional independence assumption of
traditional Naive Bayes classifiers. In particular, class-specific attribute
weighted Naive Bayes (CAWNB) has been shown to yield excellent per-
formance on many modern datasets. Such methods, however, are prone
to over-fitting on small sample, large feature space data. In this work, we
propose a Bayesian Regularized Iterative Shrinkage-Thresholding Algo-
rithm (BARISTA), which includes both ℓ1 and ℓ2 regularization to mitigate
this problem. As we show, estimating the parameters of BARISTA via
maximum likelihood yields a convex objective that can be efficiently
optimized using Iterative Shrinkage-Thresholding Algorithms (ISTA).
We prove the resulting method has many attractive theoretical and
numerical properties, including a guaranteed linear rate of convergence.
Using several standard benchmark datasets, we demonstrate how BARISTA
can yield a significant increase in performance compared to many state-
of-the-art weighted Naive Bayes methods. We also show how the Fast
Iterative-Shrinkage Thresholding Algorithm (FISTA) can be used to
further accelerate convergence.


### Data Access

We direct users to the UCI Machine Learning Repository [link](https://archive.ics.uci.edu/ml/index.php). If this work is accepted, we will release a link to our research group's one-drive that contains already pre-processed data. For now, the pre-preocessed CSV files are included in this repo. A user will need to change the file path in the experiment file to run our experiments.


### How To Run

To use BARISTA, please see the how_to_run.ipynb file that contains detailed instructions. A user will need a dataset, along with specificaions of parameter values when calling the BARISTA object. Below is some basic **documentation** about the algorith. 

class BARISTA.fit(training_samples, training_labels, scheme = 'FISTA', learning_rate = 0.1, convergence_constant = 1e-6, max_iterations= 5000, l1_penalty = 0.01, l2_penalty = 0.001)

<ins>parameters</ins>

*training_samples*: $(R^{n \times m})$ dataframe object of training data with the class attribute removed 

*training_labels*: $(R^{n})$ dataframe object of training labels that map to the training samples

*scheme*: (string) optimization scheme (either FISTA or ISTA) to learn optimal weight values, *default*: FISTA

*learning_rate*: (float) initial learning rate during the backtracking line search

*convergence_constant*: (float) tolerance used to decide when to stop iterating 

*max_iterations*: (int) maximum number of iterations allowed duing model learning

$\ell_1$ *Penalty*: (float) penalty constant for $\ell_1$ regulatization

$\ell_2$ *Penalty*: (float) penalty constant for $\ell_2$ regulatization

<ins>output</ins>

*model_weights*: ($R^{l \times m}$) Optimal set of class-dependent weights

*priors*: ($R^{l}$) prior probabilities learned from training labels

*likelihoods*: $(\Theta \in R^{l \times m})^n_{i=1}$ likelihood probabilities learned from the training samples and labels

*posteriors*: $(\hat{P}_{train} \in R^{n \times l})$ posterior distribution learned from the training samples and labels


_____________________________________________________________________________________________________________________________________________________________

class BARISTA.predict(testing_samples, testing_labels)

*testing_samples*: $(R^{k \times m})$ dataframe object of testing data with the class attribute removed

*testing_labels*: $(R^{k})$ dataframe object of testing labels that map to the testing samples

<ins>output</ins>

*predicted classification*: ($\max \hat{P}_{test} \rightarrow c \in R^{k}$) predicted classifications corresponding to the testing samples



_____________________________________________________________________________________________________________________________________________________________

class BARISTA.model_accuracy(predictions, ground_truth)

*predictions*: ($R^k$) array of predictions from BARISTA.predict

*ground_truth*: ($R^k$) testing labels that map to the testing samples

<ins>output</ins>

*accuracy metric*: ($R$) accuracy score given by $1 - \frac{|\text{errors}|}{k}$
