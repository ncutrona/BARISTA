#POSTERIOR Class

#Needed Libraries
import numpy as np
import pandas as pd
import likelihood
import prior

class PosteriorDistribution:

    def __init__(self, weight_matrix, likelihood_matrix, priors):
        
        self.weight_matrix = weight_matrix #Passing in the current weight matrix 
        self.priors = priors #Passing in the priors computed from the prior class
        self.num_classes = len(self.priors)
        self.likelihood_matrix = likelihood_matrix #Passing in the training sample likelihood matrix
        self.posterior_distribution = self.posterior_computation() #Computing PD - Complex Posterior

    #Posterior Computations
    def posterior_computation(self):

        posteriors = []
        attribute_length = len(self.weight_matrix[0])

        reg_term = 0 #Denominator Computation
        for i in range(self.num_classes):
            likelihoods = 1
            for j in range(attribute_length):
                likelihoods *= (self.likelihood_matrix[i][j] ** self.weight_matrix[i][j])
            reg_term += (self.priors[i] * likelihoods)
        
        #Computing numerator and dividing by the regularization term for each class to form the posterior
        for i in range(self.num_classes):
            likelihoods = 1
            for j in range(attribute_length):
                likelihoods *= (self.likelihood_matrix[i][j] ** self.weight_matrix[i][j])

            posterior = (self.priors[i] * likelihoods) / reg_term
            posteriors.append(posterior)
        
        return posteriors


#Test 5: Making sure complex and simple posterior_probabilities are correct
'''data = pd.DataFrame(columns = ['x1', 'x2', 'y'])
data['x1'] = [1,1,0]
data['x2'] = [1,1,0]
data['y'] = ["y", "n", "y"]
matrix = LikelihoodMatrix(data[['x1', 'x2']], data['y']).likelihood_matrices[1]
labels = ['y','n','y']
priors = Priors(labels)
vector = priors.prior_vector
posterior = PosteriorDistribution(np.ones(shape = (2,2)),np.ones(shape = (2,)), 0.5, matrix, vector )
#Test 5: Making Sure simple and complex posterior values are correct
assert round(posterior.complex_posteriors()[0],3) == 2*0.287
assert round(posterior.complex_posteriors()[1],3) == 2*0.213
assert round(posterior.simple_posteriors()[0],3) == 2*0.287
assert round(posterior.simple_posteriors()[1],3) == 2*0.213

#Test 6: Making sure all posterior distributions sum to 1 for a given observation
assert np.sum(posterior.posterior_estimations) == 1'''