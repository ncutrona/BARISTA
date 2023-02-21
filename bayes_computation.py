import numpy as np
import pandas as pd

class Bayes_Computation:
    def __init__(self, training_samples, training_labels):
        self.training_samples = training_samples
        self.training_labels = training_labels
        self.num_samples = len(self.training_samples)
        self.num_classes = len(np.unique(self.training_labels))
        self.label_values = np.unique(self.training_labels)
        self.sample_attributes = self.training_samples.columns
        self.num_attributes = len(self.sample_attributes)
        self.likelihood_collection = self.likelihood_probability_collection()
        
    def binary_computer(self, input_one, input_two):
        if(input_one == input_two): 
            return 1
        else:
            return 0

    def prior_probabilities(self):
        prior_probabilities = []
        for i in range(self.num_classes): 
            prior_sum = 0
            for j in range(self.num_samples):
                prior_sum += self.binary_computer(self.label_values[i], self.training_labels[j])
            prior = (prior_sum + (1 / self.num_classes)) / (self.num_samples + 1) 
            prior_probabilities.append(prior) 
        return prior_probabilities
    
    def likelihood_helper_one(self, attribute_value, attribute, label_value, label, laplace_constant):
        numerator_sum = 0
        for i in range(self.num_samples):
            numerator_sum += (self.binary_computer(attribute_value, attribute[i]) * self.binary_computer(label_value, label[i])) 
        numerator_sum += laplace_constant
        return numerator_sum
    
    def likelihood_helper_two(self, label_value, label):
        denominator_sum = 0
        for i in range(self.num_samples):
            denominator_sum += self.binary_computer(label_value, label[i])
        denominator_sum += 1
        return denominator_sum
    
    def likelihood_probabilitiy(self, numerator, denominator):
        return numerator/denominator 

    def likelihood_probability_collection(self):
        likelihood_probabilities = {}
        for i in range(self.num_attributes): 
            attribute_probabilities = {} 
            att_length = len(np.unique(self.training_samples[self.sample_attributes[i]])) 
            nj = 1/att_length 
            for j in range(att_length): 
                attribute_values = np.unique(self.training_samples[self.sample_attributes[i]]) 
                for p in range(self.num_classes):
                    attribute_value = attribute_values[j]
                    attribute = self.training_samples[self.sample_attributes[i]]
                    label_value = self.label_values[p]
                    label = self.training_labels
                    numerator = self.likelihood_helper_one(attribute_value, attribute, label_value, label, nj)
                    denominator = self.likelihood_helper_two(label_value, label)
                    likelihood = self.likelihood_probabilitiy(numerator, denominator) 
                    attribute_probabilities[attribute_value + label_value] = likelihood 
            likelihood_probabilities[self.sample_attributes[i]] = attribute_probabilities 
        return likelihood_probabilities

    def likelihood_matrix(self, instance):
        likelihood_matrix = []
        likelihood_probability_collection = self.likelihood_collection
        for i in range(self.num_classes):
            class_value = self.label_values[i]
            for j in range((self.num_attributes)):
                attribute_value = instance[self.sample_attributes[j]]
                likelihood_matrix.append(likelihood_probability_collection[self.sample_attributes[j]][attribute_value + class_value])
        likelihood_matrix = np.array(likelihood_matrix)
        likelihood_matrix = likelihood_matrix.reshape(self.num_classes, self.num_attributes)
        return likelihood_matrix
    
    def likelihood_probabilities(self):
        matrices = []
        for i in range(self.num_samples):
            matrices.append(self.likelihood_matrix(self.training_samples.iloc[i]))
        return matrices