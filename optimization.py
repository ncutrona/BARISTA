import numpy as np
import pandas as pd
import likelihoods
import posteriors
import priors

class Optimization:

    def __init__(self, training_samples, training_labels, max_iter, convergence_constant, lambda_term, beta_1, beta_2):
        print("_Computing Model Parameters_...\n")
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.convergence_constant = convergence_constant
        self.max_iter = max_iter
        self.lambda_term = lambda_term
        self.training_samples = training_samples
        self.training_labels = training_labels
        self.label_values = np.unique(self.training_labels)
        self.num_observations = len(self.training_labels)
        self.num_classes = len(self.label_values)
        self.priors = priors.Priors(self.training_labels).prior_vector
        self.ground_truths = self.get_ground_truth()
        self.weight_matrix = self.initialize_weight_matrix() 
        self.likelihood_storage = self.get_likelihood_storage()
        self.likelihood_matrix_cache = self.get_likelihood_cache() 
        self.M = self.model_learning()

    def initialize_weight_matrix(self):
        attribute_length = len(self.training_samples.columns)
        return np.ones(shape = (self.num_classes, attribute_length))

    def get_ground_truth(self):
        ground_truths = []
        for i in range(self.num_observations):
            truth_value = self.training_labels[i]
            ground_truth_i = []
            for j in range(self.num_classes):
                if(truth_value == self.label_values[j]):
                    ground_truth_i.append(1)
                else:
                    ground_truth_i.append(0)
            ground_truths.append(ground_truth_i)
        return ground_truths
    
    def get_likelihood_cache(self):
        return likelihoods.LikelihoodMatrix(self.training_samples, self.training_labels).likelihood_matrices

    def get_likelihood_storage(self):
        return likelihoods.LikelihoodMatrix(self.training_samples, self.training_labels).likelihood_cache
        
    def get_posterior_cache(self, weight_matrix): 
        likelihoods = self.likelihood_matrix_cache
        priors = self.priors
        posterior_distributions = []
        for i in range(self.num_observations): 
            post_object = posteriors.PosteriorDistribution(weight_matrix, likelihoods[i], priors)
            posterior_i = post_object.posterior_distribution
            posterior_distributions.append(posterior_i)
        return posterior_distributions 
    
    def objective_function(self, posterior_cache):
        scale = 1/2
        loss = 0
        for i in range(self.num_observations):
            loss_i = 0
            for j in range(self.num_classes):
                loss_i += np.square((self.ground_truths[i][j] - posterior_cache[i][j]))
            loss += loss_i
        penalty = 0
        for i in range(self.num_classes):
            for j in range(len(self.weight_matrix[0])):
                penalty+= np.abs(self.weight_matrix[i][j])
        penalty = penalty * self.lambda_term
        loss += penalty
        return scale * loss

    '''def delta_Wij(self, class_index, attribute_index, posterior_cache):
        d_sum = 0
        for i in range(self.num_observations):
            ground_truth = self.ground_truths[i][class_index]
            phat_estimation = posterior_cache[i][class_index]
            log_likelihood = np.log(self.likelihood_matrix_cache[i][class_index][attribute_index])
            d_sum += ((ground_truth - phat_estimation) * (phat_estimation * (1-phat_estimation) * log_likelihood))
        return -1 * d_sum'''

    def delta_Wij(self, class_index, attribute_index, posterior_cache):
        different_class_sum = 0
        same_class_sum = 0
        for i in range(self.num_observations):
            log_likelihood = np.log(self.likelihood_matrix_cache[i][class_index][attribute_index])
            same_class_phat_estimation = posterior_cache[i][class_index]
            same_class_ground_truth = self.ground_truths[i][class_index]
            same_class_sum += ((same_class_ground_truth - same_class_phat_estimation) * (same_class_phat_estimation * (1-same_class_phat_estimation) * log_likelihood))
            for j in range(self.num_classes):
                if(j != class_index):
                    different_class_ground_truth = self.ground_truths[i][j]
                    different_class_phat_estimation = posterior_cache[i][j]
                    different_class_sum += (different_class_ground_truth - different_class_phat_estimation) * (different_class_phat_estimation * same_class_phat_estimation * log_likelihood)
        return different_class_sum - same_class_sum


    def prox_solution(self, update, learning_rate):
        boundary = (learning_rate * self.lambda_term) / 2
        if((-1 * boundary) < update and update < boundary):
            return 0
        elif(update > boundary):
            update -= boundary
            return update
        elif(update < (-1*boundary)):
            update += boundary
            return update
        else:
            print("Proximal Descent Error - Check for Underflow/Overflow --> will return 0")
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

    def wolfe_line_search(self, gradient, posterior_cache):
        num_c = self.num_classes
        num_atts = len(self.training_samples.columns)
        alpha = 0
        t = 1
        beta = np.Inf
        notdone = True
        gradient = np.array(gradient)
        p = -1 * gradient
        max_iter = 0
        while(notdone):
            #Wolfe weights
            wolfe_posterior_cache = self.get_posterior_cache(self.weight_matrix + (t*np.reshape(p, (num_c, num_atts))))
            flat_wolfe_gradient = []
            for i in range(num_c):
                for j in range(num_atts):
                    gradient_ij = self.delta_Wij(i,j, wolfe_posterior_cache)
                    flat_wolfe_gradient.append(gradient_ij)
            if(self.objective_function(wolfe_posterior_cache) > self.objective_function(posterior_cache) + self.beta_1*t*np.dot(p, gradient)):
                beta = t
                t = 0.5*(alpha+beta)
            elif(-np.dot(p, flat_wolfe_gradient) > -1*self.beta_2*np.dot(p, gradient)):
                alpha = t
                if(beta == np.Inf):
                    t = 2*alpha
                else:
                    t = 0.5*(alpha+beta)
            else:
                notdone = False
                h = t
            if(max_iter > 50):
                notdone = False
                h = t
            max_iter += 1
        return(h)

    #Optimization function
    def model_learning(self):
        weight_matrix = np.copy(self.weight_matrix)
        posterior_cache = self.get_posterior_cache(self.weight_matrix)
        loss_values = []
        weight_collection = []
        iteration = 0
        num_c = self.num_classes
        num_atts = len(self.training_samples.columns)
        converged = False
        learning_rate = 0
        
        if(self.max_iter > 0):
            print("_Optimizing_...\n")
        
        while(converged != True and iteration < self.max_iter):
            flat_weight_matrix_gradient = []
            for i in range(num_c):
                for j in range(num_atts):
                    gradient_ij = self.delta_Wij(i,j, posterior_cache)
                    flat_weight_matrix_gradient.append(gradient_ij)
            learning_rate = self.wolfe_line_search(flat_weight_matrix_gradient, posterior_cache)
            graident_matrix = np.reshape(flat_weight_matrix_gradient, (num_c, num_atts))
            for i in range(num_c):
                for j in range(num_atts):
                    update = weight_matrix[i][j] - (learning_rate * graident_matrix[i][j])
                    proximal_solution = self.prox_solution(update, learning_rate)
                    weight_matrix[i][j] = proximal_solution
            
            weight_collection.append(np.copy(weight_matrix))
            self.weight_matrix = weight_matrix
            posterior_cache = self.get_posterior_cache(self.weight_matrix)
            gradient_norm = self.gradient_norm(flat_weight_matrix_gradient)
            loss = self.objective_function(posterior_cache)
            loss_values.append(loss)

            if(iteration > 0):
                converged = self.convergence_check(loss_values[-2], loss_values[-1])
    
            print("Iteration:", iteration+1)
            print("Wolfe's Alpha:", learning_rate)
            print("Penalty Term:", self.lambda_term)
            print("Posterior Cache First Sample:", posterior_cache[0])
            print("Weight Matrix:", self.weight_matrix)
            print("Gradient Weight Matrix Norm:", gradient_norm)
            print("Model Loss:", loss)
            print("Converged:", converged)
            print("=================================================================")
            
            iteration += 1
            
        self.loss = loss_values
        if(converged):
            print("_Optimization Successful_")
            self.posterior_cache = posterior_cache    
            return [self.weight_matrix, weight_collection]  
        else:
            print("_Optimization Failed_")
            self.posterior_cache = posterior_cache
            return [self.weight_matrix, weight_collection]