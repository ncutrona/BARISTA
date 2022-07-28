import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import framework

class CrossValidation:

    def __init__(self, k, training_samples, training_labels, max_iter, convergence_constant, lambda_parameters, beta_1, beta_2):
        self.k = k
        self.training_samples = training_samples
        self.training_labels = training_labels
        self.max_iter = max_iter
        self.convergence_constant = convergence_constant
        self.lambda_parameters = lambda_parameters
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.best_lambda = self.get_best_parameter()

    def cross_val(self, penalty):
        cross_accuracies = []
        kf = KFold(n_splits=self.k)
        for train_index, test_index in kf.split(self.training_samples):
            X_train, X_test = self.training_samples.iloc[train_index, :], self.training_samples.iloc[test_index, :]
            y_train, y_test = self.training_labels.iloc[train_index], self.training_labels.iloc[test_index]
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
            cross_accuracies.append(framework.Framework(X_train, y_train, X_test, y_test, self.max_iter, self.convergence_constant, penalty, self.beta_1, self.beta_2).test_accuracy)
        return np.mean(cross_accuracies)

    def get_best_parameter(self):
        penalty_accuracies = []
        for i in range(len(self.lambda_parameters)):
            penalty_accuracies.append(self.cross_val(self.lambda_parameters[i]))
        max_accuracy = max(penalty_accuracies)
        max_index = penalty_accuracies.index(max_accuracy)
        best_lambda = self.lambda_parameters[max_index]
        return best_lambda 