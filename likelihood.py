#LIKELIHOOD Class

#Needed Libraries
import numpy as np
import pandas as pd


class LikelihoodMatrix:

    def __init__(self, training_samples, training_labels):

        self.training_samples = training_samples #Input space of the training samples
        self.training_labels = training_labels #Label values for the training samples
        self.label_values = np.unique(self.training_labels) #List of possible label values
        self.num_classes = len(self.label_values) #Number of unique class values
        self.num_observations = len(self.training_labels) #Number of observations
        self.sample_features = self.training_samples.columns #List of sample feature names
        self.likelihood_cache = self.get_likelihood_cache() #Global likelihood cache
        self.likelihood_matrices = self.get_populated_likelihood_matrices() #Instance likelihood matrix 

    
    #Helper Functions
    def binary_computer(self, input_one, input_two):
        
        if(input_one == input_two): #Returns 1 if inputs are equivalent, 0 otherwise
            return 1
        else:
            return 0

    #Calculates numerator of the likelihood computation
    def likelihood_numerator_helper(self, attribute_value, attribute, label_value, label, num_attribute_values):
        
        numerator_sum = 0
        for i in range(self.num_observations):
            numerator_sum += (self.binary_computer(attribute_value, attribute[i]) * self.binary_computer(label_value, label[i])) 
        numerator_sum += num_attribute_values #1/num_att_vals
        
        return numerator_sum
    
    #Calculates the denominator of the likelihood computation
    def likelihood_denominator_helper(self, label_value, label):

        denominator_sum = 0
        for i in range(self.num_observations):
            denominator_sum += self.binary_computer(label_value, label[i])
        denominator_sum += 1

        return denominator_sum
    
    #Computes likelihood values
    def compute_likelihood(self, numerator, denominator):

        return numerator/denominator
    
    #Computes one global storage of all likelihoods used for taining
    def get_likelihood_cache(self):

        attributes = self.sample_features
        likelihoods = {}

        for i in range(len(attributes)): #Iterating through each attribute
            cache = {} #Attribute-wise storage unit
            att_length = len(np.unique(self.training_samples[attributes[i]])) #total number of unique values for the ith attribute
            nj = 1/att_length #Constant term
            for j in range(att_length): #Iterating through each attribute value for a given attribute
                attribute_values = np.unique(self.training_samples[attributes[i]]) #Unique attribute values
                for p in range(len(self.label_values)): #Iterating through all the class values
                    attribute_value = attribute_values[j]
                    attribute = self.training_samples[attributes[i]]
                    label_value = self.label_values[p]
                    label = self.training_labels
                    numerator = self.likelihood_numerator_helper(attribute_value, attribute, label_value, label, nj)
                    denominator = self.likelihood_denominator_helper(label_value, label)
                    likelihood = self.compute_likelihood(numerator, denominator) #Calculated Likelihood on class basis
                    cache[str(attribute_value) + str(label_value)] = likelihood #Adding likelihood value to the attribute cache
            likelihoods[attributes[i]] = cache #Adding the dictionary to the gloabl likelihood cache

        return likelihoods

            
    #Populates the likelihood matrix as shown in Wang et al.
    def populate_likelihood_matrix(self, instance):

        attributes = self.training_samples.columns 
        likelihood_matrix = []
        likelihood_cache = self.likelihood_cache
        
        #Populating the likelihood matrix by pulling the values from the global likelihood cache
        for i in range(self.num_classes):
            class_value = str(self.label_values[i])
            for j in range(len(attributes)):
                attribute_value = str(instance[attributes[j]])
                likelihood_matrix.append(likelihood_cache[attributes[j]][attribute_value + class_value])
        
        likelihood_matrix = np.array(likelihood_matrix)
        likelihood_matrix = likelihood_matrix.reshape(self.num_classes, len(attributes)) #Reshaping into a matrix
        return likelihood_matrix
    
    #Filling all observation likelihood matrices
    def get_populated_likelihood_matrices(self):

        matrices = []
        for i in range(self.num_observations):
            matrices.append(self.populate_likelihood_matrix(self.training_samples.iloc[i]))
        
        return matrices


'''#Test two: Making sure likelihood matrix outputs the correct values
data = pd.DataFrame(columns = ['x1', 'x2', 'y'])
data['x1'] = [1,1,0]
data['x2'] = [1,1,0]
data['y'] = ["y", "n", "y"]
cache = LikelihoodMatrix(data[['x1', 'x2']], data['y'])
np.testing.assert_array_equal(cache.likelihood_matrices[0], np.array([[0.75,0.75],[0.5,0.5]]))

#Test three: Making sure likelihood numerator outputs the correct values
av = 1
a = [1, 1, 0]
label_val = 'y'
label = ['y','n','y']
nj = 1/len(np.unique(a))
assert cache.likelihood_numerator_helper(av, a, label_val, label, nj) == 1.5

#Test Four: Making sure likelihood deniminator outputs the correct values
label_val = 'y'
label = ['y','n','y']
assert cache.likelihood_denominator_helper(label_val, label) == 3'''