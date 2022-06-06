#PRIOR Class

#Needed Libraries
import numpy as np
import pandas as pd

#The Priors class stores the prior probabilities
class Priors:

    #Constructor
    def __init__(self, training_labels):

        self.training_labels = training_labels #Label values for prior vector computation
        self.label_values = np.unique(self.training_labels) #Class values for the labels
        self.num_classes = len(self.label_values) #Number of Classes in the data set
        self.num_observations = len(self.training_labels) #Number of instances in the data set
        self.prior_vector = self.compute_priors() #Computing the prior vector
    
    #Helper Function
    def binary_computer(self, input_one, input_two):
        
        if(input_one == input_two): #Returns 1 if inputs are equivalent, 0 otherwise
            return 1
        else:
            return 0

    #Prior vector computation
    def compute_priors(self):

        prior_vect = [] #Prior vector to be returned
        for i in range(self.num_classes): #Iterate through each label value
            prior_sum = 0
            for j in range(self.num_observations): #Iterate through each observation to include in the sum 
                prior_sum += self.binary_computer(self.label_values[i], self.training_labels[j])
            prior = (prior_sum + (1 / self.num_classes)) / (self.num_observations + 1) #Compute prior as (13) in Wang et al.
            prior_vect.append(prior) #Add computed prior to the prior vector
        
        return prior_vect #Return prior vector


#Test One: Making sure prior outputs the correct values
'''labels = ['y','y','n']
priors = Priors(labels)
vector = priors.prior_vector
assert vector == [0.375, 0.625]'''
 