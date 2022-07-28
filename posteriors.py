import numpy as np
import pandas as pd

class PosteriorDistribution:

    def __init__(self, weight_matrix, likelihood_matrix, priors):
        self.weight_matrix = weight_matrix  
        self.priors = priors 
        self.num_classes = len(self.priors)
        self.likelihood_matrix = likelihood_matrix 
        self.posterior_distribution = self.posterior_computation()

    def posterior_computation(self):
        posteriors = []
        numerators = []
        attribute_length = len(self.weight_matrix[0])
        reg_term = 0 
        for i in range(self.num_classes):
            likelihoods = 1
            for j in range(attribute_length):
                likelihoods *= (self.likelihood_matrix[i][j] ** self.weight_matrix[i][j])
            numerator = self.priors[i] * likelihoods
            numerators.append(numerator)
            reg_term += numerator
        for i in range(len(numerators)):
            posterior = numerators[i] / reg_term
            if(np.isnan(posterior)):
                posteriors.append(np.finfo(float).eps) #Handles underflow
            else:
                posteriors.append(posterior)
        return posteriors