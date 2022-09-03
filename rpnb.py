import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import cross_validation
import framework
import preprocessing

class RPNB:
    
    def __init__(self, k, dataframe, target_attribute, test_split, continous_attributes, max_iter, convergence_constant, lambda_parameters, beta_1, beta_2):
        self.k = k
        self.dataframe = dataframe
        self.target_attribute = target_attribute
        self.continuous_attributes = continous_attributes
        self.test_split = test_split
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.preproc = preprocessing.Preprocessing(self.dataframe, self.target_attribute, self.test_split, self.continuous_attributes)
        self.training_samples, self.training_labels, self.testing_samples, self.testing_labels = self.preproc.training_samples, self.preproc.training_labels, self.preproc.testing_samples, self.preproc.testing_labels
        self.max_iter = max_iter
        self.convergence_constant = convergence_constant
        self.lambda_parameters = lambda_parameters
        self.best_lambda = self.get_best_lambda()

    def get_best_lambda(self):
        return cross_validation.CrossValidation(self.k, self.training_samples, self.training_labels, self.max_iter, self.convergence_constant, self.lambda_parameters, self.beta_1, self.beta_2).best_lambda

    def ten_fold_simulation(self):
        cross_accuracies = []
        kf = KFold(n_splits = 10)
        target = self.dataframe[self.target_attribute]
        samples = self.dataframe.drop(self.target_attribute, axis = 1)
        for train_index, test_index in kf.split(samples):
            X_train, X_test = samples.iloc[train_index, :], samples.iloc[test_index, :]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
            cross_accuracies.append(framework.Framework(X_train, y_train, X_test, y_test, self.max_iter, self.convergence_constant, self.best_lambda, self.beta_1, self.beta_2).test_accuracy)
        return np.mean(cross_accuracies)

    def final_model(self):
        print("\n__Optimized Model__")
        model = framework.Framework(self.training_samples, self.training_labels, self.testing_samples, self.testing_labels, self.max_iter, self.convergence_constant, self.best_lambda, self.beta_1, self.beta_2)
        return model

    def plot_results(self, model):
        print("Testing Accuracy: ", model.test_accuracy)
        plt.plot(model.training_loss)
        plt.title("RPNB Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")