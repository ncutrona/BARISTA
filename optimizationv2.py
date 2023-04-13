import numpy as np
import pandas as pd
import bayes_computation
import numdifftools as nd

class Optimization:

    def __init__(self, learning_rate, training_samples, training_labels, max_iter, convergence_constant, penalty):
        self.learning_rate = learning_rate
        self.convergence_constant = convergence_constant
        self.max_iter = max_iter
        self.penalty = penalty
        
        self.training_samples = training_samples
        self.training_labels = training_labels
        self.num_samples = len(self.training_labels)
        self.num_classes = len(np.unique(self.training_labels))
        self.label_values = np.unique(self.training_labels)
        self.sample_attributes = self.training_samples.columns
        self.num_attributes = len(self.sample_attributes)
        
        self.bayes_object = bayes_computation.Bayes_Computation(self.training_samples, self.training_labels)
        self.prior_probabilities = self.bayes_object.prior_probabilities()
        print("Prior Probability Distribution Computed...")
        self.likelihood_probabilities = self.bayes_object.likelihood_probabilities()
        print("Prior Probability Distribution Computed...")
        
        self.weights = self.initialize_weight_matrix()
        self.ground_truths = self.get_ground_truth()
        self.posterior_probabability_distribution = self.posterior_probabilities()
        print("Initial Posterior Probability Distribution Computed...")
        self.M = self.model_learning()


    def initialize_weight_matrix(self):
        return np.ones(shape = (self.num_classes, self.num_attributes))

    def get_ground_truth(self):
        ground_truths = []
        for i in range(self.num_samples):
            truth_value = self.training_labels[i]
            ground_truth_i = []
            for j in range(self.num_classes):
                if(truth_value == self.label_values[j]):
                    ground_truth_i.append(1)
                else:
                    ground_truth_i.append(0)
            ground_truths.append(ground_truth_i)
        return ground_truths
    
    def regularized_posterior(self, likelihood_matrix, weights):
        posteriors = []
        numerators = []
        regularization_term = 0 
        for i in range(self.num_classes):
            likelihoods = 1
            for j in range(self.num_attributes):
                likelihoods *= (likelihood_matrix[i][j] ** weights[i][j])
            numerator = self.prior_probabilities[i] * likelihoods
            numerators.append(numerator)
            regularization_term += numerator
        for i in range(len(numerators)):
            posterior = numerators[i] / regularization_term
            if(np.isnan(posterior)):
                posteriors.append(np.finfo(float).eps) 
            else:
                posteriors.append(posterior)
        return posteriors

    def posterior_probabilities(self): 
        posterior_distributions = []
        for i in range(self.num_samples): 
            posterior = self.regularized_posterior(self.likelihood_probabilities[i], self.weights)
            posterior_distributions.append(posterior)
        return posterior_distributions 
    
    def differentiable_objective_term(self):
        scale = 1/2
        loss = 0
        for i in range(self.num_samples):
            loss_i = 0
            for j in range(self.num_classes):
                loss_i += np.square((self.ground_truths[i][j] - self.posterior_probabability_distribution[i][j]))
            loss += loss_i       
        return scale * loss 

    def non_differentiable_objective_term(self):
        return np.sum(np.abs(self.weights.flatten()))
    
    def model_loss(self,):
        return self.differentiable_objective_term() + self.non_differentiable_objective_term()

    def grad_Wij(self, class_index, attribute_index):
        different_class_sum = 0
        same_class_sum = 0
        for i in range(self.num_samples):
            log_likelihood = np.log(self.likelihood_probabilities[i][class_index][attribute_index])
            same_class_phat_estimation = self.posterior_probabability_distribution[i][class_index]
            same_class_ground_truth = self.ground_truths[i][class_index]
            same_class_sum += ((same_class_ground_truth - same_class_phat_estimation) * (same_class_phat_estimation * (1-same_class_phat_estimation) * log_likelihood))
            for j in range(self.num_classes):
                if(j != class_index):
                    different_class_ground_truth = self.ground_truths[i][j]
                    different_class_phat_estimation = self.posterior_probabability_distribution[i][j]
                    different_class_sum += (different_class_ground_truth - different_class_phat_estimation) * (different_class_phat_estimation * same_class_phat_estimation * log_likelihood)
        return different_class_sum - same_class_sum

    def soft_thresholding(self, update, learning_rate):
        boundary = (learning_rate * self.penalty) / 2
        if((-1 * boundary) < update and update < boundary):
            return 0
        elif(update > boundary):
            update -= boundary
            return update
        elif(update < (-1*boundary)):
            update += boundary
            return update
        else:
            print("Proximal Descent Error --> will return 0")
            return 0 

    def convergence_check(self, loss_old, loss_new):
        converged = False
        convergence_check = np.abs((loss_old - loss_new)/max(np.abs(loss_old), np.abs(loss_new), 1)) 
        if(convergence_check < self.convergence_constant):
            converged = True
            return converged
        else:
            return converged
    
    def gradient_norm(self, gradient):
        grad_norm = round(np.sqrt(np.sum(np.square(gradient))), 6)
        return grad_norm

    def model_learning(self):
        print("__Optimizing__...")
        weights = np.copy(self.weights)
        loss_values = []
        weight_collection = []
        iteration = 1
        converged = False
        loss = self.model_loss()
        loss_values.append(loss)
        #Hessian_obj = nd.Hessian(self.obj_func)
        #min_eig_values = []
        
        gradient_matrix = np.zeros((self.num_classes, self.num_attributes))
        while(converged != True and iteration < self.max_iter):
            learning_rate = np.copy(self.learning_rate)
            while(learning_rate > 0.0005):
                flag = False
                current_weights = np.copy(weights)
                for i in range(self.num_classes):
                    for j in range(self.num_attributes):
                        gradient_ij = self.grad_Wij(i,j)
                        gradient_matrix[i][j] = gradient_ij
                        gd_update = current_weights[i][j] - (learning_rate * gradient_matrix[i][j])
                        proximal_update = self.soft_thresholding(gd_update, learning_rate)
                        current_weights[i][j] = proximal_update
                
                self.weights = current_weights
                self.posterior_probabability_distribution = self.posterior_probabilities()
                loss = self.model_loss()

                if(loss > loss_values [-1]):
                    learning_rate *= 0.5
                else:
                    flag = True
                    break

            if(flag == False):
                learning_rate = learning_rate * 2
            weights = np.copy(self.weights)
            weight_collection.append(self.weights)
            gradient_norm = self.gradient_norm(self.weights.flatten())
            loss_values.append(loss)
            converged = self.convergence_check(loss_values[-2], loss_values[-1])

            #Convexity Analysis, Hessian PSD
            #if(iteration%20 == 0):
                #H = Hessian_obj(weight_matrix.flatten())
                #min_eig = min(np.linalg.eig(H)[0])
                #min_eig_values.append(min_eig)

            '''print("Iteration:", iteration)
            print("Learning Rate:", learning_rate)
            print("Penalty Term:", self.penalty)
            print("Posterior Cache First Sample:", self.posterior_probabability_distribution[0])
            print("Weight Matrix:", self.weights)
            print("Gradient Weight Matrix Norm:", gradient_norm)
            print("Model Loss:", loss)'''
            #if(iteration%20 ==0):
                #print("Min Eigenvalue:", min_eig)
            #print("Converged:", converged)
            #print("=================================================================")
            
            iteration += 1
            
        self.loss = loss_values
        if(converged):
            print("_Optimization Successful_")  
            return [self.weights, weight_collection]  
        else:
            print("_Optimization Failed_")
            return [self.weights, weight_collection]
        
