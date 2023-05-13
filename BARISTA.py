import numpy as np
import pandas as pd
import optimization


class BARISTA:

    def __init__(self):
        pass

    def fit(self, training_samples, training_labels, scheme, learning_rate, convergence_constant, max_iterations, l1_penalty, l2_penalty):
        self.training_samples = training_samples
        self.training_labels = training_labels
        self.scheme = scheme
        self.learning_rate = learning_rate
        self.convergence_constant = convergence_constant
        self.max_iterations = max_iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        
        print("Computing Model Parameters...")
        self.model_training = optimization.Optimization(self.training_samples, self.training_labels,  self.scheme, self. learning_rate, self.max_iterations, self.convergence_constant, self.l1_penalty, self.l2_penalty)
        self.num_samples = self.model_training.training_samples
        self.num_classes = self.model_training.num_classes
        self.label_values = self.model_training.label_values
        self.sample_attributes = self.model_training.sample_attributes
        self.num_attributes = self.model_training.num_attributes
        self.optimal_weights = self.model_training.M[0]
        self.weight_collection = self.model_training.M[1]
        self.prior_probabilities = self.model_training.prior_probabilities
        self.likelihood_probabilities = self.model_training.likelihood_probabilities
        self.likelihood_collection = self.model_training.bayes_object.likelihood_collection

    def predict(self, testing_samples, testing_labels):
        self.testing_samples = testing_samples
        self.testing_labels = testing_labels
        self.num_testing_samples = len(self.testing_labels)
        self.testing_likelihoods = self.testing_sample_likelihood_probabilities()
        self.posterior_distribution = self.fit_posteriors()
        predictions = []
        for i in range(self.num_testing_samples):
            max_index = self.posterior_distribution[i].index(max(self.posterior_distribution[i]))
            prediction = self.label_values[max_index]
            predictions.append(prediction)
        self.predictions = predictions
        self.accuracy = self.model_accuracy(self.predictions, self.testing_labels)
        
    def get_testing_sample_likelihood(self, test_sample):
        likelihood_matrix = []
        for i in range(self.num_classes):
            class_value = self.label_values[i]
            for j in range(self.num_attributes):
                attribute_value = test_sample[self.sample_attributes[j]]
                try:
                    likelihood_matrix.append(self.likelihood_collection[self.sample_attributes[j]][attribute_value + class_value])
                except KeyError:
                    nj = 1 / (len(np.unique(self.training_samples[self.sample_attributes[j]])))
                    c = self.training_labels.value_counts()
                    c_val = c[class_value] + 1
                    likelihood_matrix.append(nj/c_val)
        likelihood_matrix = np.array(likelihood_matrix)
        likelihood_matrix = likelihood_matrix.reshape(self.num_classes, self.num_attributes)
        return likelihood_matrix
    
    def testing_sample_likelihood_probabilities(self):
        likeihood_probabilities = []
        for i in range(self.num_testing_samples):
            likeihood_probabilities.append(self.get_testing_sample_likelihood(self.testing_samples.iloc[i]))
        return likeihood_probabilities

    def fit_posteriors(self): 
        testing_sample_likelihoods = self.testing_sample_likelihood_probabilities()
        testing_posterior_distributions = []
        for i in range(self.num_testing_samples): 
            posterior_i = self.model_training.regularized_posterior(testing_sample_likelihoods[i], self.optimal_weights)
            testing_posterior_distributions.append(posterior_i)
        return testing_posterior_distributions

    def model_accuracy(self, predictions, ground_truth):
        errors = 0
        for i in range(len(predictions)):
            if(predictions[i] != ground_truth[i]):
                errors += 1
        return 1 - (errors/len(predictions))
    
    def norm_differnces(self):
        differences = []
        for i in range(len(self.weight_collection)):
            differences.append(np.linalg.norm(self.weight_collection[i] - self.optimal_weights, 'fro'))
        return differences