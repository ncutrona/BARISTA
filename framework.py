#Framework Class

#Needed Libraries
import numpy as np
import pandas as pd
import Optimization

class Framework:

    def __init__(self, training_samples, training_labels, testing_samples, testing_labels, max_iter, convergence_constant, learning_rate, lambda_term):

        self.training_samples = training_samples
        self.training_labels = training_labels
        self.testing_samples = testing_samples
        self.testing_labels = testing_labels
        
        self.max_iter = max_iter
        self.convergence_constant = convergence_constant
        self.learning_rate = learning_rate
        self.lambda_term = lambda_term
        
        self.num_observations = len(self.training_labels)
        self.num_classes = len(np.unique(self.training_labels))
        self.num_test_observations = len(self.testing_labels)
        self.label_values = np.unique(self.training_labels)
        
        #Optimized Model Parameters
        self.optimized_model = Optimization(self.training_samples, self.training_labels, self.max_iter, self.convergence_constant, self.learning_rate, self.lambda_term)
        self.M = self.optimized_model.M
        self.priors = self.optimized_model.priors
        self.likelihoods = self.optimized_model.likelihood_storage
        self.train_posteriors = self.optimized_model.posterior_cache

        #Fitted testing data likelihoods
        self.fitted_likelihoods = self.populate_test_matrices()
        self.fitted_posteriors = self.fit_posteriors()
        
        #Results
        self.test_predictions = self.predict(self.fitted_posteriors, self.num_test_observations)
        self.train_predictions = self.predict(self.train_posteriors, self.num_observations)
        self.test_accuracy = self.model_accuracy(self.test_predictions, self.testing_labels)
        self.train_accuracy = self.model_accuracy(self.train_predictions, self.training_labels)


    def fit_likelihoods(self, instance):

        likelihood_cache = self.likelihoods
        attributes = self.training_samples.columns 
        likelihood_matrix = []
        
        #Populating the likelihood matrix by pulling the values from the global likelihood cache
        for i in range(self.num_classes):
            class_value = str(self.label_values[i])
            for j in range(len(attributes)):
                attribute_value = str(instance[attributes[j]])
                try:
                    likelihood_matrix.append(likelihood_cache[attributes[j]][attribute_value + class_value])
                except KeyError:
                    nj = 1 / (len(np.unique(self.training_samples[attributes[j]])))
                    c = self.training_labels.value_counts()
                    c_val = c[class_value] + 1
                    likelihood_matrix.append(nj/c_val)
        
        likelihood_matrix = np.array(likelihood_matrix)
        likelihood_matrix = likelihood_matrix.reshape(self.num_classes, len(attributes)) #Reshaping into a matrix
        return likelihood_matrix
    
    def populate_test_matrices(self):
        
        matrices = []
        for i in range(self.num_test_observations):
            matrices.append(self.fit_likelihoods(self.testing_samples.iloc[i]))
        
        return matrices

    #Getting the cache of observation posterior distributions
    def fit_posteriors(self): #Will be called each time the weight matrix is updated.

        fitted_likelihoods = self.fitted_likelihoods
        priors = self.priors
        posterior_distributions = []
        for i in range(self.num_test_observations): #Creating a list of posteriors for each sample 
            post_object = PosteriorDistribution(self.M[0], fitted_likelihoods[i], priors)
            posterior_i = post_object.posterior_distribution
            posterior_distributions.append(posterior_i)

        return posterior_distributions #Returns the posterior distribution, as well as the simple and complex distributions


    def predict(self, posteriors, observation_count):

        class_values = self.label_values
        predictions = []

        for i in range(observation_count):
            max_index = posteriors[i].index(max(posteriors[i]))
            prediction = class_values[max_index]
            predictions.append(prediction)

        return predictions

    def model_accuracy(self, predictions, ground_truth):

        errors = 0
        for i in range(len(predictions)):
            if(predictions[i] != ground_truth[i]):
                errors += 1
        
        return 1 - (errors/len(predictions))
        