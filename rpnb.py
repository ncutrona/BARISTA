import numpy as np
import pandas as pd
import cross_validation
import framework

class RPNB:
    
    def __init__(self, k, dataframe, target_attribute, test_split, continous_attributes, max_iter, convergence_constant, lambda_parameters, beta_1, beta_2):
        self.k = k
        self.dataframe = dataframe
        self.target_attribute = target_attribute
        self.continuous_attributes = continous_attributes
        self.test_split = test_split
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.preproc = Preprocessing(self.dataframe, self.target_attribute, self.test_split, self.continuous_attributes)
        self.training_samples, self.training_labels, self.testing_samples, self.testing_labels = self.preproc.training_samples, self.preproc.training_labels, self.preproc.testing_samples, self.preproc.testing_labels
        self.max_iter = max_iter
        self.convergence_constant = convergence_constant
        self.lambda_parameters = lambda_parameters
        self.best_lambda = CrossValidation(self.k, self.training_samples, self.training_labels, self.max_iter, self.convergence_constant, self.lambda_parameters, self.beta_1, self.beta_2).best_lambda
        self.model = self.final_model()

    def final_model(self):
        print("\n__Optimized Model__")
        model = Framework(self.training_samples, self.training_labels, self.testing_samples, self.testing_labels, self.max_iter, self.convergence_constant, self.best_lambda, self.beta_1, self.beta_2)
        return model

    def plot_results(self):
      print("Testing Accuracy: ", self.model.test_accuracy)
      plt.plot(self.model.training_loss)
      plt.title("RPNB Training Loss")
      plt.xlabel("Iteration")
      plt.ylabel("Loss")