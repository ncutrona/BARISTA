import numpy as np
import pandas as pd

class LikelihoodMatrix:

    def __init__(self, training_samples, training_labels):
        self.training_samples = training_samples
        self.training_labels = training_labels
        self.label_values = np.unique(self.training_labels)
        self.num_classes = len(self.label_values)
        self.num_observations = len(self.training_labels)
        self.sample_features = self.training_samples.columns
        self.likelihood_cache = self.get_likelihood_cache()
        self.likelihood_matrices = self.get_populated_likelihood_matrices() 

    def binary_computer(self, input_one, input_two):
        if(input_one == input_two): 
            return 1
        else:
            return 0

    def likelihood_numerator_helper(self, attribute_value, attribute, label_value, label, laplace_constant):
        numerator_sum = 0
        for i in range(self.num_observations):
            numerator_sum += (self.binary_computer(attribute_value, attribute[i]) * self.binary_computer(label_value, label[i])) 
        numerator_sum += laplace_constant
        return numerator_sum

    def likelihood_denominator_helper(self, label_value, label):
        denominator_sum = 0
        for i in range(self.num_observations):
            denominator_sum += self.binary_computer(label_value, label[i])
        denominator_sum += 1
        return denominator_sum

    def compute_likelihood(self, numerator, denominator):
        return numerator/denominator

    def get_likelihood_cache(self):
        attributes = self.sample_features
        likelihoods = {}
        for i in range(len(attributes)): 
            cache = {} 
            att_length = len(np.unique(self.training_samples[attributes[i]])) 
            nj = 1/att_length 
            for j in range(att_length): 
                attribute_values = np.unique(self.training_samples[attributes[i]]) 
                for p in range(self.num_classes):
                    attribute_value = attribute_values[j]
                    attribute = self.training_samples[attributes[i]]
                    label_value = self.label_values[p]
                    label = self.training_labels
                    numerator = self.likelihood_numerator_helper(attribute_value, attribute, label_value, label, nj)
                    denominator = self.likelihood_denominator_helper(label_value, label)
                    likelihood = self.compute_likelihood(numerator, denominator) 
                    cache[str(attribute_value) + str(label_value)] = likelihood 
            likelihoods[attributes[i]] = cache 
        return likelihoods

    def populate_likelihood_matrix(self, instance):
        attributes = self.training_samples.columns 
        likelihood_matrix = []
        likelihood_cache = self.likelihood_cache
        for i in range(self.num_classes):
            class_value = str(self.label_values[i])
            for j in range(len(attributes)):
                attribute_value = str(instance[attributes[j]])
                likelihood_matrix.append(likelihood_cache[attributes[j]][attribute_value + class_value])
        likelihood_matrix = np.array(likelihood_matrix)
        likelihood_matrix = likelihood_matrix.reshape(self.num_classes, len(attributes)) #Reshaping into a matrix
        return likelihood_matrix
    
    def get_populated_likelihood_matrices(self):
        matrices = []
        for i in range(self.num_observations):
            matrices.append(self.populate_likelihood_matrix(self.training_samples.iloc[i]))
        return matrices