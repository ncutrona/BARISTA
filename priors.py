import numpy as np
import pandas as pd

class Priors:

    def __init__(self, training_labels):
        self.training_labels = training_labels 
        self.label_values = np.unique(self.training_labels) 
        self.num_classes = len(self.label_values) 
        self.num_observations = len(self.training_labels) 
        self.prior_vector = self.compute_priors() 

    def binary_computer(self, input_one, input_two):
        if(input_one == input_two): 
            return 1
        else:
            return 0

    def compute_priors(self):
        prior_vect = [] 
        for i in range(self.num_classes): 
            prior_sum = 0
            for j in range(self.num_observations):
                prior_sum += self.binary_computer(str(self.label_values[i]), str(self.training_labels[j])) #Handle digits as character representations (Example Treat 1 as '1')
            prior = (prior_sum + (1 / self.num_classes)) / (self.num_observations + 1) 
            prior_vect.append(prior) 
        return prior_vect