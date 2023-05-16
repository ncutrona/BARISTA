# Bayesian Regularized Iterative Soft Thresholding Algorithm


The BARISTA repository is a dedicated codebase to work recently submitted to the NeurIPS 2023 Conference. This readme file will contain the paper abstract, a link to the data used for experimentation, and a python notebook with dedicated instructions on how to use our algorithm.


### Paper Abstract

Weighted Naive Bayes methods have recently been developed to alleviate the strong conditional independence assumption of traditional Naive Bayes classifiers. In particular, class-specific attribute weighted Naive Bayes (CAWNB) has been shown to yield excellent performance on many modern datasets. Such methods, however, are prone to over-fitting on small sample, large feature space data. In this work, we propose a Bayesian Regularized Iterative Shrinkage-Thresholding Algorithm *BARISTA*, which includes both $\ell_1$ and $\ell_2$ regularization to mitigate this problem. As we show, estimating the parameters of *BARISTA* via maximum likelihood yields a convex objective that can be efficiently optimized using Iterative shrinkage-thresholding Algorithms (ISTA). The resulting method has many attractive theoretical and numerical properties, including a guaranteed linear rate of convergence. Using several benchmark datasets, we demonstrate how *BARISTA* can yield a significant increase in performance compared to many state-of-the-art weighted Naive Bayes methods. We also show how the Fast Iterative-Shrinkage Thresholding algorithm (FISTA) can be used to further accelerate convergence.


### Data Access

For now, we direct users to the UCI Machine Learning Repository [link](https://archive.ics.uci.edu/ml/index.php). If this work is accepted, we will release a link to our research group's one-drive that contains already pre-processed data.


### How To Run

To use BARISTA, please see the how_to_run.ipyn file that contains detailed instructions. A user will need a dataset, along with specificaions of parameter values when calling the BARISTA object. Below is some basic **documentation** about the algorith. 





