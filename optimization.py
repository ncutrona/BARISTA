import numpy as np
import pandas as pd
import bayes_computation
import numdifftools as nd
import matplotlib.pyplot as plt

class Optimization:

    def __init__(self, training_samples, training_labels, scheme, learning_rate, max_iter, convergence_constant, l1_penalty, l2_penalty):
        self.scheme = scheme
        self.learning_rate = learning_rate
        self.convergence_constant = convergence_constant
        self.max_iter = max_iter
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        
        self.training_samples = training_samples
        self.training_labels = training_labels
        self.num_samples = len(self.training_labels)
        self.num_classes = len(np.unique(self.training_labels))
        self.label_values = np.unique(self.training_labels)
        self.sample_attributes = self.training_samples.columns
        self.num_attributes = len(self.sample_attributes)
        self.gradient_matrix = np.ones((self.num_classes, self.num_attributes))
        
        self.bayes_object = bayes_computation.Bayes_Computation(self.training_samples, self.training_labels)
        self.prior_probabilities = self.bayes_object.prior_probabilities()
        print("Prior Probability Distribution Computed...")
        self.likelihood_probabilities = self.bayes_object.likelihood_probabilities()
        print("Prior Probability Distribution Computed...")
        
        self.weights = self.initialize_weight_matrix()
        self.ground_truths = self.get_ground_truth()
        self.posterior_probabability_distribution = self.posterior_probabilities()
        print("Initial Posterior Probability Distribution Computed...")
        if(self.scheme == "ISTA"):
            print("ISTA Scheme Selected")
            self.M = self.ISTA()
        else:
            print("FISTA Scheme Selected (Default)")
            self.M = self.FISTA()


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
            posteriors = self.regularized_posterior(self.likelihood_probabilities[i], self.weights)
            posterior_distributions.append(posteriors)
        return posterior_distributions 
    
    def model_loss(self,):
        log_likelihood_sum = 0
        for i in range(self.num_samples):
            sample_truth_index = self.ground_truths[i].index(max(self.ground_truths[i]))
            sample_posterior = self.posterior_probabability_distribution[i][sample_truth_index]
            if(sample_posterior == 0):
                 sample_posterior += np.finfo(float).eps
            log_likelihood_sum += np.log(sample_posterior)
        return -1 * log_likelihood_sum +  self.l1_penalty*sum(abs(self.weights.reshape(-1))) + self.l2_penalty*sum((self.weights.reshape(-1))**2)

    def gradient(self, weights):
        l2 = self.l2_penalty * 2*(weights)
        E = np.zeros(shape = (self.num_classes, self.num_attributes))
        for i in range(self.num_samples):
            sample_truth_index = self.ground_truths[i].index(max(self.ground_truths[i]))
            for k in range(self.num_attributes):
                #C' = C
                log_likelihood = np.log(self.likelihood_probabilities[i][sample_truth_index][k])
                E[sample_truth_index][k] += (-1 * log_likelihood)
                #C' != C and C' = C
                for c in range(self.num_classes):
                    log_likelihood = np.log(self.likelihood_probabilities[i][c][k])
                    sample_posterior_c = self.posterior_probabability_distribution[i][c]
                    E[c][k] += (sample_posterior_c * log_likelihood)       
        return ((1/self.num_samples) * (E)) + l2

    def soft_thresholding(self, update, learning_rate):
        boundary = learning_rate * self.l1_penalty
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
        
    def convergence_check(self, weight_1, weight_2):
        converged = False
        convergence_criterion = np.sum(np.abs((weight_1 - weight_2)).flatten())
        if(convergence_criterion < self.convergence_constant):
            converged = True
        return converged

    def gradient_norm(self, gradient):
        grad_norm = np.linalg.norm(gradient, 'fro')
        return grad_norm

    def quad_approx(self, x, y, fy, learning_rate, gradient):
        l1 = self.l1_penalty*sum(abs(x.reshape(-1)))
        Q = fy + (x-y).reshape(-1).dot(gradient.reshape(-1)) + 1/(2*learning_rate)*np.linalg.norm(x-y)**2 + l1
        return(Q)

    def FISTA(self):
        print("__Optimizing__...")
        weights = np.copy(self.weights)
        loss_values = []
        weight_collection = [weights]
        iteration = 1
        converged = False
        loss = self.model_loss()
        print("Initial Loss:", loss)
        loss_values.append(loss)
        fista_weights = [weights]
        t = 1
        #Hessian_obj = nd.Hessian(self.obj_func)
        #min_eig_values = []
        
        while(converged != True and iteration <= self.max_iter):
            learning_rate = np.copy(self.learning_rate)
            while(learning_rate > 1e-8):
                current_weights = np.copy(fista_weights[-1])
                self.gradient_matrix = self.gradient(current_weights)
                current_weights = current_weights - (learning_rate * self.gradient_matrix)
                for i in range(self.num_classes):
                    for j in range(self.num_attributes):
                        current_weights[i][j] = self.soft_thresholding(current_weights[i][j], learning_rate)

                self.weights = current_weights
                self.posterior_probabability_distribution = self.posterior_probabilities()
                loss = self.model_loss()

                previous_loss_no_penalty = loss_values[-1]-self.l1_penalty*sum(abs(fista_weights[-1].reshape(-1)))
                if(loss < self.quad_approx(current_weights, fista_weights[-1], previous_loss_no_penalty, learning_rate, self.gradient_matrix)):
                    t_new = (1 + np.sqrt(1 + 4*(t**2)))/2
                    fista_weights.append(current_weights + ((t -1)/t_new)*(current_weights - weight_collection[-1]))
                    t = t_new
                    break
                else:
                    learning_rate *= 0.5
                    
            
            weights = np.copy(self.weights)
            weight_collection.append(self.weights)
            gradient_norm = self.gradient_norm(self.gradient_matrix)
            loss_values.append(loss)
            converged = self.convergence_check(weight_collection[-1], weight_collection[-2])

            #Convexity Analysis, Hessian PSD
            #if(iteration%20 == 0):
                #H = Hessian_obj(weight_matrix.flatten())
                #min_eig = min(np.linalg.eig(H)[0])
                #min_eig_values.append(min_eig)

            print("Iteration:", iteration)
            print("Learning Rate:", learning_rate)
            print("L1_Penalty Term:", self.l1_penalty)
            print("L2_Penalty Term:", self.l2_penalty)
            print("Posterior Cache First Sample:", self.posterior_probabability_distribution[0])
            print("\nWeight Matrix:", self.weights)
            print("\nGradient Norm:", gradient_norm)
            print("\nGradient Matrix:", self.gradient_matrix)
            print("\nModel Loss:", loss)
            print("==============================================================================================================================\n")
            #if(iteration%20 ==0):
                #print("Min Eigenvalue:", min_eig)
            #print("Converged:", converged)
            #print("=================================================================")
            
            iteration += 1
            
        self.loss = loss_values
        if(converged):
            print("_Optimization Successful_")
            #plt.plot(loss_values)  
            #plt.show()
            return [self.weights, weight_collection]  
        else:
            print("_Optimization Failed_")
            #plt.plot(loss_values)
            #plt.show()
            return [self.weights, weight_collection]
        

    def ISTA(self):
        print("__Optimizing__...")
        weights = np.copy(self.weights)
        loss_values = []
        weight_collection = [weights]
        iteration = 1
        converged = False
        loss = self.model_loss()
        print("Initial Loss:", loss)
        loss_values.append(loss)
        #Hessian_obj = nd.Hessian(self.obj_func)
        #min_eig_values = []
        
        while(converged != True and iteration <= self.max_iter):
            learning_rate = np.copy(self.learning_rate)
            while(learning_rate > 1e-8):
                current_weights = np.copy(weights)
                self.gradient_matrix = self.gradient(current_weights)
                current_weights = current_weights - (learning_rate * self.gradient_matrix)
                for i in range(self.num_classes):
                    for j in range(self.num_attributes):
                        current_weights[i][j] = self.soft_thresholding(current_weights[i][j], learning_rate)

                self.weights = current_weights
                self.posterior_probabability_distribution = self.posterior_probabilities()
                loss = self.model_loss()

                previous_loss_no_penalty = loss_values[-1]-self.l1_penalty*sum(abs(weights.reshape(-1)))
                if(loss < self.quad_approx(current_weights, weights, previous_loss_no_penalty, learning_rate, self.gradient_matrix)):
                    break
                else:
                    learning_rate *= 0.5
                    
            
            weights = np.copy(self.weights)
            weight_collection.append(self.weights)
            gradient_norm = self.gradient_norm(self.gradient_matrix)
            loss_values.append(loss)
            converged = self.convergence_check(weight_collection[-1], weight_collection[-2])

            #Convexity Analysis, Hessian PSD
            #if(iteration%20 == 0):
                #H = Hessian_obj(weight_matrix.flatten())
                #min_eig = min(np.linalg.eig(H)[0])
                #min_eig_values.append(min_eig)

            print("Iteration:", iteration)
            print("Learning Rate:", learning_rate)
            print("L1_Penalty Term:", self.l1_penalty)
            print("L2_Penalty Term:", self.l2_penalty)
            print("Posterior Cache First Sample:", self.posterior_probabability_distribution[0])
            print("\nWeight Matrix:", self.weights)
            print("\nGradient Norm:", gradient_norm)
            print("\nGradient Matrix:", self.gradient_matrix)
            print("\nModel Loss:", loss)
            print("==============================================================================================================================\n")
            #if(iteration%20 ==0):
                #print("Min Eigenvalue:", min_eig)
            #print("Converged:", converged)
            #print("=================================================================")
            
            iteration += 1
            
        self.loss = loss_values
        if(converged):
            print("_Optimization Successful_")
            #plt.plot(loss_values)  
            #plt.show()
            return [self.weights, weight_collection]  
        else:
            print("_Optimization Failed_")
            #plt.plot(loss_values)
            #plt.show()
            return [self.weights, weight_collection]







