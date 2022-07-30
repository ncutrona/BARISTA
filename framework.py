import numpy as np
import pandas as pd
import optimization
import posteriors

class Framework:

    def __init__(self, training_samples, training_labels, testing_samples, testing_labels, max_iter, convergence_constant, lambda_term, beta_1, beta_2):
        self.training_samples = training_samples
        self.training_labels = training_labels
        self.testing_samples = testing_samples
        self.testing_labels = testing_labels
        self.max_iter = max_iter
        self.convergence_constant = convergence_constant
        self.lambda_term = lambda_term
        self.num_observations = len(self.training_labels)
        self.num_classes = len(np.unique(self.training_labels))
        self.num_test_observations = len(self.testing_labels)
        self.label_values = np.unique(self.training_labels)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.optimized_model = optimization.Optimization(self.training_samples, self.training_labels, self.max_iter, self.convergence_constant, self.lambda_term, self.beta_1, self.beta_2)
        self.M = self.optimized_model.M
        self.training_loss = self.optimized_model.loss
        self.priors = self.optimized_model.priors
        self.likelihoods = self.optimized_model.likelihood_storage
        self.train_posteriors = self.optimized_model.posterior_cache
        self.fitted_likelihoods = self.populate_test_matrices()
        self.fitted_posteriors = self.fit_posteriors()
        self.test_predictions = self.predict(self.fitted_posteriors, self.num_test_observations)
        self.train_predictions = self.predict(self.train_posteriors, self.num_observations)
        self.test_accuracy = self.model_accuracy(self.test_predictions, self.testing_labels)
        self.train_accuracy = self.model_accuracy(self.train_predictions, self.training_labels)

    def fit_likelihoods(self, instance):
        likelihood_cache = self.likelihoods
        attributes = self.training_samples.columns 
        likelihood_matrix = []
        for i in range(self.num_classes):
            class_value = self.label_values[i]
            for j in range(len(attributes)):
                attribute_value = instance[attributes[j]]
                try:
                    likelihood_matrix.append(likelihood_cache[attributes[j]][attribute_value + class_value])
                except KeyError:
                    nj = 1 / (len(np.unique(self.training_samples[attributes[j]])))
                    c = self.training_labels.value_counts()
                    c_val = c[class_value] + 1
                    likelihood_matrix.append(nj/c_val)
        likelihood_matrix = np.array(likelihood_matrix)
        likelihood_matrix = likelihood_matrix.reshape(self.num_classes, len(attributes))
        return likelihood_matrix
    
    def populate_test_matrices(self):
        matrices = []
        for i in range(self.num_test_observations):
            matrices.append(self.fit_likelihoods(self.testing_samples.iloc[i]))
        return matrices

    def fit_posteriors(self): 
        fitted_likelihoods = self.fitted_likelihoods
        priors = self.priors
        posterior_distributions = []
        for i in range(self.num_test_observations): 
            post_object = posteriors.PosteriorDistribution(self.M[0], fitted_likelihoods[i], priors)
            posterior_i = post_object.posterior_distribution
            posterior_distributions.append(posterior_i)
        return posterior_distributions

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
        